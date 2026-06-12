"""
xP  –  Expected Pass Model
===========================
Infers pass events from ball-carrier changes in SNMOT tracking data,
extracts geometric and contextual features, trains a Logistic Regression
model (leave-one-sequence-out CV), and writes per-sequence CSV files to
output/<seq>_xp.csv mirroring the existing MP4 output structure.

Features
--------
Geometric (fully derived from bounding boxes):
  x_start          – normalised foot-x of passer  [0, 1]
  y_start_mirrored – normalised foot-y, mirrored for left/right symmetry
  angle_deg        – passing angle relative to attack direction (°)
  distance_px      – Euclidean pixel distance between feet

Contextual (heuristic approximations):
  is_cross         – 1 if passer is in wide zone and receiver moves inside
  is_header        – 1 if ball is in upper 30 % of carrier bbox
  is_set_piece     – always 0 (cannot be inferred from positions)
  is_ground        – always 1 (ball height not in GT data)

Labels
------
  outcome = 1  →  carrier changes to a teammate  (completed pass)
  outcome = 0  →  carrier changes to an opponent  (intercepted / incomplete)

Usage
-----
  python xp_model.py               # all sequences
  python xp_model.py --seq SNMOT-070
"""

import argparse
import configparser
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
TRAIN_DIR  = Path(__file__).parent / "train"
OUTPUT_DIR = Path(__file__).parent / "output"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Pixels – sequences are 1920 × 1080
IMG_W = 1920
IMG_H = 1080

# Minimum frames between two carrier-change events (debounce)
MIN_GAP_FRAMES = 3

# Wide-zone thresholds for cross detection (fraction of image height)
CROSS_WIDE_ZONE   = 0.20   # passer must be within 20 % of top/bottom edge
CROSS_CENTRE_ZONE = 0.60   # receiver must land in central 60 %

# Ball centre in upper fraction of bbox → header
HEADER_THRESHOLD = 0.30

# CSV columns
CSV_FIELDS = [
    "sequence", "frame_start", "frame_end",
    "passer_tid", "receiver_tid",
    "x_start", "y_start_mirrored",
    "angle_deg", "distance_px",
    "is_cross", "is_header", "is_set_piece", "is_ground",
    "outcome", "xP",
]

FEATURE_COLS = [
    "x_start", "y_start_mirrored",
    "angle_deg", "distance_px",
    "is_cross", "is_header", "is_set_piece", "is_ground",
]

# ---------------------------------------------------------------------------
# Parsing helpers (shared with main.py logic)
# ---------------------------------------------------------------------------

def parse_gameinfo(path: Path):
    """Return team_map {tid -> 'left'|'right'|None}, ball_id."""
    team_map = {}
    ball_id  = None
    cfg = configparser.ConfigParser()
    cfg.read(path)
    if "Sequence" not in cfg:
        return team_map, ball_id
    for key, value in cfg["Sequence"].items():
        if not key.startswith("trackletid_"):
            continue
        try:
            idx = int(key[len("trackletid_"):])
        except ValueError:
            continue
        value = value.strip()
        role  = value.split(";")[0].strip().lower()
        if "ball" in role and "ball boy" not in role:
            ball_id       = idx
            team_map[idx] = None
        elif any(x in role for x in ("referee", "crowd", "ball boy")):
            team_map[idx] = None
        elif "left" in role:
            team_map[idx] = "left"
        elif "right" in role:
            team_map[idx] = "right"
        else:
            team_map[idx] = None
    return team_map, ball_id


def parse_gt(path: Path):
    """Return {frame_id: {track_id: (x, y, w, h)}}."""
    fd = defaultdict(dict)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            p = line.split(",")
            if len(p) < 6:
                continue
            fid = int(p[0]);  tid = int(p[1])
            fd[fid][tid] = (float(p[2]), float(p[3]), float(p[4]), float(p[5]))
    return fd


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def foot_point(x, y, w, h):
    return (x + w / 2.0, y + h)


def centre_point(x, y, w, h):
    return (x + w / 2.0, y + h / 2.0)


def dist2(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def find_carrier(ball_pos, player_bboxes: dict):
    """Return tid of player whose foot is closest to ball_pos."""
    best_tid, best_d = None, float("inf")
    for tid, bbox in player_bboxes.items():
        d = dist2(ball_pos, foot_point(*bbox))
        if d < best_d:
            best_d, best_tid = d, tid
    return best_tid


# ---------------------------------------------------------------------------
# Feature extraction for a single pass event
# ---------------------------------------------------------------------------

def extract_features(
    passer_bbox, receiver_bbox, ball_bbox,
    passer_side,            # 'left' or 'right'
    img_w=IMG_W, img_h=IMG_H,
):
    px, py, pw, ph = passer_bbox
    rx, ry, rw, rh = receiver_bbox

    passer_foot   = foot_point(px, py, pw, ph)
    receiver_foot = foot_point(rx, ry, rw, rh)

    # Normalised start position
    x_start = passer_foot[0] / img_w
    y_start = passer_foot[1] / img_h

    # Mirror y so both sides look symmetric:
    # left-team attacks rightwards → y_mirrored = y_start
    # right-team attacks leftwards → y_mirrored = 1 - y_start
    y_start_mirrored = y_start if passer_side == "left" else (1.0 - y_start)

    # Angle (degrees). Forward = positive x for left-team, negative for right-team.
    dx = receiver_foot[0] - passer_foot[0]
    dy = receiver_foot[1] - passer_foot[1]
    if passer_side == "right":
        dx = -dx  # flip so forward = positive x
    angle_deg = math.degrees(math.atan2(dy, dx))

    # Distance
    distance_px = math.hypot(
        receiver_foot[0] - passer_foot[0],
        receiver_foot[1] - passer_foot[1],
    )

    # is_cross: passer in wide zone, receiver more central
    passer_y_norm = passer_foot[1] / img_h
    recvr_y_norm  = receiver_foot[1] / img_h
    in_wide  = passer_y_norm < CROSS_WIDE_ZONE or passer_y_norm > (1 - CROSS_WIDE_ZONE)
    in_centre = CROSS_WIDE_ZONE < recvr_y_norm < (1 - CROSS_WIDE_ZONE)
    # also check the receiver is somewhat forward (toward goal)
    is_cross = int(in_wide and in_centre)

    # is_header: ball centre in upper 30 % of carrier's bbox
    is_header = 0
    if ball_bbox is not None:
        ball_cy = centre_point(*ball_bbox)[1]
        bbox_top    = py
        bbox_height = ph
        rel = (ball_cy - bbox_top) / bbox_height if bbox_height > 0 else 0.5
        is_header = int(rel < HEADER_THRESHOLD)

    is_set_piece = 0  # cannot determine from tracking positions
    is_ground    = 1  # default assumption

    return {
        "x_start":          round(x_start, 4),
        "y_start_mirrored": round(y_start_mirrored, 4),
        "angle_deg":        round(angle_deg, 2),
        "distance_px":      round(distance_px, 2),
        "is_cross":         is_cross,
        "is_header":        is_header,
        "is_set_piece":     is_set_piece,
        "is_ground":        is_ground,
    }


# ---------------------------------------------------------------------------
# Pass detection for one sequence
# ---------------------------------------------------------------------------

def detect_passes(frame_data, team_map, ball_id, seq_name):
    """
    Walk through frames in order.  Each time the ball carrier changes,
    record a pass event dict.

    Returns list of dicts (without xP – to be filled later).
    """
    frames = sorted(frame_data.keys())
    passes = []

    prev_carrier_tid  = None
    prev_carrier_side = None
    prev_frame        = None
    last_event_frame  = -MIN_GAP_FRAMES - 1

    for fid in frames:
        cur = frame_data[fid]

        # Ball position
        ball_bbox = cur.get(ball_id) if ball_id else None
        if ball_bbox is None:
            continue
        ball_pos = centre_point(*ball_bbox)

        # All field players
        left_p  = {t: b for t, b in cur.items() if team_map.get(t) == "left"}
        right_p = {t: b for t, b in cur.items() if team_map.get(t) == "right"}
        all_p   = {**left_p, **right_p}
        if not all_p:
            continue

        # Find overall carrier (closest player)
        carrier_tid = find_carrier(ball_pos, all_p)
        carrier_side = team_map.get(carrier_tid)

        # Detect carrier change
        if (carrier_tid != prev_carrier_tid
                and prev_carrier_tid is not None
                and (fid - last_event_frame) >= MIN_GAP_FRAMES):

            # Only log pass if previous carrier was a real field player
            if prev_carrier_side in ("left", "right"):
                # Determine outcome
                if carrier_side == prev_carrier_side:
                    outcome = 1   # completed (teammate received)
                elif carrier_side in ("left", "right"):
                    outcome = 0   # intercepted (opponent received)
                else:
                    # Ball went to unknown/referee/out – skip
                    prev_carrier_tid  = carrier_tid
                    prev_carrier_side = carrier_side
                    prev_frame        = fid
                    continue

                # Get bboxes at transition frame
                passer_bbox   = frame_data.get(prev_frame, {}).get(prev_carrier_tid)
                receiver_bbox = cur.get(carrier_tid)

                if passer_bbox and receiver_bbox:
                    feats = extract_features(
                        passer_bbox, receiver_bbox,
                        frame_data.get(prev_frame, {}).get(ball_id),
                        prev_carrier_side,
                    )
                    event = {
                        "sequence":    seq_name,
                        "frame_start": prev_frame,
                        "frame_end":   fid,
                        "passer_tid":  prev_carrier_tid,
                        "receiver_tid": carrier_tid,
                        **feats,
                        "outcome": outcome,
                        "xP": None,  # filled after training
                    }
                    passes.append(event)
                    last_event_frame = fid

        prev_carrier_tid  = carrier_tid
        prev_carrier_side = carrier_side
        prev_frame        = fid

    return passes


# ---------------------------------------------------------------------------
# ML: train + predict
# ---------------------------------------------------------------------------

def train_and_predict(all_passes: list[dict], target_seq: str):
    """
    Leave-one-sequence-out: train on all sequences EXCEPT target_seq,
    predict xP for target_seq passes.

    Returns xP values (list of floats) for target_seq passes in order.
    """
    train_rows = [p for p in all_passes if p["sequence"] != target_seq]
    test_rows  = [p for p in all_passes if p["sequence"] == target_seq]

    if not test_rows:
        return []

    def to_X(rows):
        return np.array([[r[c] for c in FEATURE_COLS] for r in rows], dtype=float)

    X_test = to_X(test_rows)

    if not train_rows:
        # No training data – return 0.5 as neutral prior
        return [0.5] * len(test_rows)

    X_train = to_X(train_rows)
    y_train = np.array([r["outcome"] for r in train_rows], dtype=int)

    # Need both classes to train a meaningful model
    if len(set(y_train)) < 2:
        prior = float(y_train.mean())
        return [prior] * len(test_rows)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_s, y_train)

    proba = model.predict_proba(X_test_s)
    # Column index for class=1 (complete)
    cls1_idx = list(model.classes_).index(1)
    return [round(float(p[cls1_idx]), 4) for p in proba]


# ---------------------------------------------------------------------------
# Per-sequence processing
# ---------------------------------------------------------------------------

def process_sequence(seq_dir: Path):
    name        = seq_dir.name
    gt_path     = seq_dir / "gt"  / "gt.txt"
    gameinfo_p  = seq_dir / "gameinfo.ini"

    if not gt_path.exists():
        print(f"  [{name}] SKIP – missing gt.txt")
        return name, []

    team_map, ball_id = parse_gameinfo(gameinfo_p)
    frame_data        = parse_gt(gt_path)
    passes            = detect_passes(frame_data, team_map, ball_id, name)
    print(f"  [{name}] {len(passes)} pass events detected")
    return name, passes


# ---------------------------------------------------------------------------
# Write CSV
# ---------------------------------------------------------------------------

def write_csv(passes: list[dict], output_dir: Path, seq_name: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{seq_name}_xp.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(passes)
    return out_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="xP – Expected Pass Model")
    parser.add_argument("--seq", nargs="*", metavar="SNMOT-XXX")
    args = parser.parse_args()

    if not TRAIN_DIR.exists():
        sys.exit(f"[ERROR] train/ not found: {TRAIN_DIR}")

    all_seqs = sorted(d for d in TRAIN_DIR.iterdir()
                      if d.is_dir() and d.name.startswith("SNMOT"))
    if not all_seqs:
        sys.exit("[ERROR] No sequences found in train/")

    selected = ([TRAIN_DIR / s for s in args.seq] if args.seq else all_seqs)
    for s in selected:
        if not s.exists():
            sys.exit(f"[ERROR] Not found: {s}")

    print(f"xP Model  –  {len(selected)} sequence(s)")
    print("=" * 60)

    # ── Step 1: detect passes in ALL sequences (needed for LOSO CV) ──
    print("\n[1/3] Detecting passes across all sequences …")
    seq_passes: dict[str, list[dict]] = {}
    for seq_dir in all_seqs:              # always use full set for training
        name, passes = process_sequence(seq_dir)
        seq_passes[name] = passes

    all_passes = [p for ps in seq_passes.values() for p in ps]
    n_complete   = sum(1 for p in all_passes if p["outcome"] == 1)
    n_incomplete = sum(1 for p in all_passes if p["outcome"] == 0)
    print(f"\n  Total passes: {len(all_passes)}"
          f"  (complete={n_complete}, intercepted={n_incomplete})")

    # ── Step 2: LOSO prediction ──
    print("\n[2/3] Training logistic regression (leave-one-sequence-out) …")
    for seq_dir in selected:
        name = seq_dir.name
        xp_vals = train_and_predict(all_passes, name)
        target_passes = seq_passes[name]
        for p, xp in zip(target_passes, xp_vals):
            p["xP"] = xp
        print(f"  [{name}] xP computed for {len(xp_vals)} passes")

    # ── Step 3: write CSVs ──
    print("\n[3/3] Writing CSVs …")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for seq_dir in selected:
        name = seq_dir.name
        passes = seq_passes[name]
        if not passes:
            print(f"  [{name}] no passes – skipping CSV")
            continue
        out_path = write_csv(passes, OUTPUT_DIR, name)
        print(f"  [{name}] → {out_path}")

    print(f"\n✅  Done. CSVs → {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()

"""
Passing Polygon CV Pipeline  —  Ball-carrier mode
==================================================
For every SNMOT-* sequence in train/:
  1. Parse gameinfo.ini  → track_id → team ('left'|'right') or None
                         → ball track_id
  2. Parse gt/gt.txt     → per-frame bounding boxes (all tracks)
  3. For each frame:
       a. Load real match image from img1/
       b. Find the ball position
       c. Find the field player closest to the ball  → BALL CARRIER
       d. Draw ALL players' bounding boxes + foot-node dots
       e. Highlight the ball carrier with a special ring
       f. Draw the ball icon
       g. Draw passing lines ONLY from ball carrier → each teammate:
              GREEN  = open (no opponent in the path)
              RED    = blocked (opponent bbox intersects path)
  4. Write annotated frames to output/<seq>_passing.mp4

Usage:
    python main.py                     # all sequences
    python main.py --seq SNMOT-070    # single sequence
"""

import argparse
import configparser
import math
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend – no display needed
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# xP utilities shared with xp_model.py
from xp_model import (
    parse_gameinfo  as _xp_parse_gameinfo,
    parse_gt        as _xp_parse_gt,
    detect_passes   as _xp_detect_passes,
    extract_features as _xp_extract_features,
    FEATURE_COLS    as _XP_FEATURE_COLS,
)

# ---------------------------------------------------------------------------
# xP model  (global – trained once before the first sequence)
# ---------------------------------------------------------------------------
_XP_MODEL:  LogisticRegression | None = None
_XP_SCALER: StandardScaler     | None = None
_XP_CLS1_IDX: int = 1          # index for class=1 in predict_proba

def build_xp_model():
    """Train a global Logistic Regression on ALL sequences in train/."""
    global _XP_MODEL, _XP_SCALER, _XP_CLS1_IDX
    print("\n[xP] Building global model …")
    all_passes = []
    for seq_dir in sorted(TRAIN_DIR.iterdir()):
        if not seq_dir.is_dir() or not seq_dir.name.startswith("SNMOT"):
            continue
        gt_path = seq_dir / "gt" / "gt.txt"
        gi_path = seq_dir / "gameinfo.ini"
        if not gt_path.exists():
            continue
        tm, bid = _xp_parse_gameinfo(gi_path)
        fd      = _xp_parse_gt(gt_path)
        all_passes.extend(_xp_detect_passes(fd, tm, bid, seq_dir.name))

    if len(all_passes) < 10 or len({p["outcome"] for p in all_passes}) < 2:
        print("[xP] Not enough data – model skipped (xP will default to 0.50)")
        return

    X = np.array([[p[c] for c in _XP_FEATURE_COLS] for p in all_passes], dtype=float)
    y = np.array([p["outcome"] for p in all_passes], dtype=int)

    scaler = StandardScaler()
    X_s    = scaler.fit_transform(X)
    model  = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_s, y)

    _XP_MODEL    = model
    _XP_SCALER   = scaler
    _XP_CLS1_IDX = list(model.classes_).index(1)
    print(f"[xP] Model ready  ({len(all_passes)} passes, "
          f"complete={sum(y)}, intercepted={len(y)-sum(y)})")


def compute_xp(passer_bbox, receiver_bbox, ball_bbox, passer_side) -> float:
    """Return xP ∈ [0,1] for one candidate pass."""
    if _XP_MODEL is None or _XP_SCALER is None:
        return 0.50
    feats = _xp_extract_features(passer_bbox, receiver_bbox, ball_bbox, passer_side)
    X     = np.array([[feats[c] for c in _XP_FEATURE_COLS]], dtype=float)
    X_s   = _XP_SCALER.transform(X)
    return round(float(_XP_MODEL.predict_proba(X_s)[0][_XP_CLS1_IDX]), 2)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
TRAIN_DIR  = Path(__file__).parent / "train"
OUTPUT_DIR = Path(__file__).parent / "output"

# ---------------------------------------------------------------------------
# Visual constants  (BGR)
# ---------------------------------------------------------------------------
HOME_BOX_COLOUR     = (210, 140,  30)   # amber  – team left
AWAY_BOX_COLOUR     = (30,   30, 220)   # red    – team right
HOME_NODE_COLOUR    = (255, 220, 100)
AWAY_NODE_COLOUR    = (120, 120, 255)

CARRIER_RING_COLOUR = (0,   255, 255)   # bright yellow ring around ball carrier
BALL_COLOUR         = (50,  230, 255)   # bright yellow dot for ball

OPEN_LINE_COLOUR    = (50,  230,  50)   # green  – open passing lane
BLOCK_LINE_COLOUR   = (30,   30, 220)   # red    – blocked lane

LABEL_COLOUR        = (255, 255, 255)

# Sizes
NODE_RADIUS          = 5
CARRIER_RING_RADIUS  = 14
BALL_RADIUS          = 6
OPEN_LINE_THICKNESS  = 2
BLOCK_LINE_THICKNESS = 2
LINE_ALPHA           = 0.75   # blend factor for the line overlay

# Opponent bbox expansion for block detection (px)
BLOCK_EXPAND_PX = 12

# Minimum distance (px) for a passing line to be drawn
MIN_PASS_DIST = 30

# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_gameinfo(path: Path):
    """
    Returns:
        team_map   : {track_id -> 'left'|'right'|None}
        ball_id    : int | None   (track_id of the ball)
        jersey_map : {track_id -> jersey_str}
    """
    team_map   = {}
    jersey_map = {}
    ball_id    = None

    cfg = configparser.ConfigParser()
    cfg.read(path)
    if "Sequence" not in cfg:
        return team_map, ball_id, jersey_map

    for key, value in cfg["Sequence"].items():
        if not key.startswith("trackletid_"):
            continue
        try:
            idx = int(key[len("trackletid_"):])
        except ValueError:
            continue
        value = value.strip()
        role  = value.split(";")[0].strip().lower()
        parts = value.split(";")
        jersey_map[idx] = parts[1].strip() if len(parts) > 1 else "?"

        if "ball" in role and "ball boy" not in role:
            ball_id       = idx
            team_map[idx] = None          # ball is not a player team
        elif any(x in role for x in ("referee", "crowd", "ball boy")):
            team_map[idx] = None
        elif "left" in role:
            team_map[idx] = "left"
        elif "right" in role:
            team_map[idx] = "right"
        else:
            team_map[idx] = None

    return team_map, ball_id, jersey_map


def parse_gt(path: Path):
    """Return frame_data[frame_id][track_id] = (x, y, w, h)."""
    fd = defaultdict(dict)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            p = line.split(",")
            if len(p) < 6:
                continue
            fid = int(p[0])
            tid = int(p[1])
            fd[fid][tid] = (float(p[2]), float(p[3]), float(p[4]), float(p[5]))
    return fd


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def foot_point(x, y, w, h):
    return (x + w / 2.0, y + h)


def centre_point(x, y, w, h):
    return (x + w / 2.0, y + h / 2.0)


def dist2(a, b):
    return (a[0]-b[0])**2 + (a[1]-b[1])**2


def segments_intersect(p1, p2, p3, p4):
    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    d1, d2 = cross(p3, p4, p1), cross(p3, p4, p2)
    d3, d4 = cross(p1, p2, p3), cross(p1, p2, p4)
    return (((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and
            ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)))


def bbox_edges(x, y, w, h, expand=0):
    x1, y1 = x - expand, y - expand
    x2, y2 = x + w + expand, y + h + expand
    return [
        ((x1,y1),(x2,y1)), ((x2,y1),(x2,y2)),
        ((x2,y2),(x1,y2)), ((x1,y2),(x1,y1)),
    ]


def is_blocked(src, dst, opponents, expand=BLOCK_EXPAND_PX):
    for (ox, oy, ow, oh) in opponents:
        for edge in bbox_edges(ox, oy, ow, oh, expand):
            if segments_intersect(src, dst, edge[0], edge[1]):
                return True
        # dst inside opponent box
        if (ox-expand) <= dst[0] <= (ox+ow+expand) and \
           (oy-expand) <= dst[1] <= (oy+oh+expand):
            return True
    return False


# ---------------------------------------------------------------------------
# Ball-carrier detection
# ---------------------------------------------------------------------------

def find_ball_carrier(ball_pos, players: dict):
    """
    Return the (tid, bbox) of the player whose foot-point is closest
    to ball_pos.  Returns (None, None) if no players present.
    """
    if not players:
        return None, None
    best_tid  = None
    best_dist = float("inf")
    for tid, bbox in players.items():
        fp   = foot_point(*bbox)
        d    = dist2(ball_pos, fp)
        if d < best_dist:
            best_dist = d
            best_tid  = tid
    return best_tid, players[best_tid]


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def draw_pass_line(canvas, src, dst, colour, thickness, xp_val: float | None = None):
    """Thick line + fixed-size arrowhead + optional xP label at midpoint."""
    cv2.line(canvas, src, dst, colour, thickness, cv2.LINE_AA)
    # arrowhead placed at 2/3 along the line
    tip = (
        src[0] + (dst[0]-src[0])*2//3,
        src[1] + (dst[1]-src[1])*2//3,
    )
    seg_len = math.hypot(tip[0]-src[0], tip[1]-src[1])
    tip_length = 12.0 / seg_len if seg_len > 0 else 0.3
    cv2.arrowedLine(canvas, src, tip, colour, thickness, cv2.LINE_AA, tipLength=tip_length)

    # xP label above the midpoint of the line
    if xp_val is not None:
        mx = (src[0] + dst[0]) // 2
        my = (src[1] + dst[1]) // 2
        label = f"xP {xp_val:.2f}"
        fs, th_l = 0.38, 1
        (tw, th_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, th_l)
        # perpendicular offset: push label ~10 px above the line
        angle = math.atan2(dst[1]-src[1], dst[0]-src[0])
        ox = int(-math.sin(angle) * 10)
        oy = int( math.cos(angle) * 10)
        lx, ly = mx + ox - tw//2, my + oy + th_h//2
        cv2.rectangle(canvas, (lx-2, ly-th_h-2), (lx+tw+2, ly+2), (0,0,0), -1)
        cv2.putText(canvas, label, (lx, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, colour, th_l, cv2.LINE_AA)


def clamp_pt(pt, w, h):
    return (int(max(0, min(w-1, pt[0]))), int(max(0, min(h-1, pt[1]))))


def draw_player_overlays(
    frame_img,
    frame_id,
    frame_data,
    team_map,
    jersey_map,
    ball_id,
    running_xp: dict | None = None,
):
    if running_xp is None:
        running_xp = {}
    cur     = frame_data.get(frame_id, {})
    H, W    = frame_img.shape[:2]

    # Separate entities
    left_players  = {tid: bbox for tid, bbox in cur.items() if team_map.get(tid) == "left"}
    right_players = {tid: bbox for tid, bbox in cur.items() if team_map.get(tid) == "right"}
    all_players   = {**left_players, **right_players}

    # Ball position
    ball_pos   = None
    ball_bbox  = cur.get(ball_id) if ball_id else None
    if ball_bbox:
        ball_pos = centre_point(*ball_bbox)

    # Find ball carrier
    carrier_tid  = None
    carrier_side = None
    if ball_pos:
        tid_l, _ = find_ball_carrier(ball_pos, left_players)
        tid_r, _ = find_ball_carrier(ball_pos, right_players)
        if tid_l is not None and tid_r is not None:
            dl = dist2(ball_pos, foot_point(*left_players[tid_l]))
            dr = dist2(ball_pos, foot_point(*right_players[tid_r]))
            if dl <= dr:
                carrier_tid, carrier_side = tid_l, "left"
            else:
                carrier_tid, carrier_side = tid_r, "right"
        elif tid_l is not None:
            carrier_tid, carrier_side = tid_l, "left"
        elif tid_r is not None:
            carrier_tid, carrier_side = tid_r, "right"

    # ── 1. Passing lines on blend overlay (only from ball carrier) ──
    line_overlay = frame_img.copy()

    if carrier_tid is not None:
        carrier_bbox = all_players[carrier_tid]
        carrier_fp   = foot_point(*carrier_bbox)
        src = clamp_pt(carrier_fp, W, H)

        # Teammates = same side, excluding carrier
        if carrier_side == "left":
            teammates = {t: b for t, b in left_players.items()  if t != carrier_tid}
            opponents = list(right_players.values())
        else:
            teammates = {t: b for t, b in right_players.items() if t != carrier_tid}
            opponents = list(left_players.values())

        for tid_t, bbox_t in teammates.items():
            tm_fp = foot_point(*bbox_t)
            dst   = clamp_pt(tm_fp, W, H)
            d     = math.hypot(src[0]-dst[0], src[1]-dst[1])
            if d < MIN_PASS_DIST:
                continue
            blocked = is_blocked(carrier_fp, tm_fp, opponents)
            colour  = BLOCK_LINE_COLOUR if blocked else OPEN_LINE_COLOUR
            thick   = BLOCK_LINE_THICKNESS if blocked else OPEN_LINE_THICKNESS
            # Compute real-time xP for this candidate pass
            xp_val = compute_xp(carrier_bbox, bbox_t, ball_bbox, carrier_side)
            draw_pass_line(line_overlay, src, dst, colour, thick, xp_val=xp_val)

    cv2.addWeighted(line_overlay, LINE_ALPHA, frame_img, 1.0 - LINE_ALPHA, 0, frame_img)

    # ── 2. All player bounding boxes + foot nodes ──
    def draw_players(players, box_col, node_col):
        for tid, (x, y, w, h) in players.items():
            x1 = max(0, int(x));      y1 = max(0, int(y))
            x2 = min(W-1, int(x+w)); y2 = min(H-1, int(y+h))
            cv2.rectangle(frame_img, (x1, y1), (x2, y2), box_col, 2)

            fp = clamp_pt(foot_point(x, y, w, h), W, H)
            cv2.circle(frame_img, fp, NODE_RADIUS+2, (0,0,0),  -1)
            cv2.circle(frame_img, fp, NODE_RADIUS,   node_col, -1)

            jersey = jersey_map.get(tid, "?")
            label  = f"#{jersey}"
            fs, th_l = 0.42, 1
            (tw, th_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, th_l)
            lx, ly = x1, max(th_h+2, y1-2)
            cv2.rectangle(frame_img, (lx, ly-th_h-2), (lx+tw+4, ly), box_col, -1)
            cv2.putText(frame_img, label, (lx+2, ly-1),
                        cv2.FONT_HERSHEY_SIMPLEX, fs, LABEL_COLOUR, th_l, cv2.LINE_AA)

    draw_players(left_players,  HOME_BOX_COLOUR, HOME_NODE_COLOUR)
    draw_players(right_players, AWAY_BOX_COLOUR, AWAY_NODE_COLOUR)

    # ── 3. Ball carrier highlight ring ──
    if carrier_tid is not None:
        cb = all_players[carrier_tid]
        fp = clamp_pt(foot_point(*cb), W, H)
        cv2.circle(frame_img, fp, CARRIER_RING_RADIUS+2, (0,0,0),            2)
        cv2.circle(frame_img, fp, CARRIER_RING_RADIUS,   CARRIER_RING_COLOUR, 2)
        # "BALL | cumxP: X.XX" label above box
        bx1 = max(0, int(cb[0]))
        by1 = max(0, int(cb[1]))
        cum_label = f"BALL | cumxP:{running_xp.get(carrier_tid, 0.0):.2f}"
        cv2.putText(frame_img, cum_label, (bx1, max(12, by1-14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, CARRIER_RING_COLOUR, 2, cv2.LINE_AA)

    # ── 4. Ball dot ──
    if ball_pos:
        bp = clamp_pt(ball_pos, W, H)
        cv2.circle(frame_img, bp, BALL_RADIUS+2, (0,0,0),      -1)
        cv2.circle(frame_img, bp, BALL_RADIUS,   BALL_COLOUR,  -1)

    # ── 5. HUD legend ──
    legend = [
        ("HOME (team left)",  HOME_BOX_COLOUR),
        ("AWAY (team right)", AWAY_BOX_COLOUR),
        ("Open pass (xP shown)",   OPEN_LINE_COLOUR),
        ("Blocked pass (xP shown)", BLOCK_LINE_COLOUR),
        ("Ball carrier (cumxP)",   CARRIER_RING_COLOUR),
    ]
    ly = 20
    for text, col in legend:
        cv2.rectangle(frame_img, (8, ly-12), (24, ly+4), col, -1)
        cv2.putText(frame_img, text, (30, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255,255,255), 1, cv2.LINE_AA)
        ly += 18


# ---------------------------------------------------------------------------
# xP chart generator
# ---------------------------------------------------------------------------

def generate_xp_chart(
    cumxp_history: dict,   # {tid: [(frame, cumxp), ...]}
    jersey_map: dict,
    team_map: dict,
    seq_name: str,
    output_dir: Path,
    total_frames: int,
):
    """Save a cumulative-xP line chart as <seq>_xp_chart.png."""
    left_tids  = [t for t, s in team_map.items() if s == "left"  and t in cumxp_history]
    right_tids = [t for t, s in team_map.items() if s == "right" and t in cumxp_history]

    # Only include players who actually had at least one pass
    left_tids  = [t for t in left_tids  if cumxp_history[t][-1][1] > 0]
    right_tids = [t for t in right_tids if cumxp_history[t][-1][1] > 0]

    if not left_tids and not right_tids:
        return  # nothing to plot

    fig, axes = plt.subplots(
        1, 2, figsize=(14, 5), sharey=False,
        gridspec_kw={"wspace": 0.35}
    )
    fig.suptitle(f"Cumulative xP per player  —  {seq_name}",
                 fontsize=13, fontweight="bold")

    palette_left  = plt.cm.autumn(np.linspace(0.15, 0.85, max(1, len(left_tids))))
    palette_right = plt.cm.winter (np.linspace(0.15, 0.85, max(1, len(right_tids))))

    for ax, tids, palette, team_label in [
        (axes[0], left_tids,  palette_left,  "HOME (team left)"),
        (axes[1], right_tids, palette_right, "AWAY (team right)"),
    ]:
        for i, tid in enumerate(sorted(tids)):
            history = cumxp_history[tid]           # [(frame, cumxp)]
            frames   = [h[0] for h in history]
            vals     = [h[1] for h in history]
            jersey   = jersey_map.get(tid, str(tid))
            ax.step(frames, vals, where="post",
                    color=palette[i], linewidth=1.8,
                    label=f"#{jersey}")

        ax.set_title(team_label, fontsize=10)
        ax.set_xlabel("Frame", fontsize=9)
        ax.set_ylabel("Cumulative xP", fontsize=9)
        ax.set_xlim(0, total_frames)
        ax.legend(fontsize=7, ncol=2, loc="upper left")
        ax.grid(True, alpha=0.3)
        ax.set_facecolor("#111111")

    fig.patch.set_facecolor("#1a1a1a")
    for ax in axes:
        ax.tick_params(colors="white", labelsize=8)
        ax.title.set_color("white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444444")
        legend = ax.get_legend()
        if legend:
            legend.get_frame().set_facecolor("#2a2a2a")
            for text in legend.get_texts():
                text.set_color("white")

    fig.suptitle(f"Cumulative xP per player  —  {seq_name}",
                 fontsize=13, fontweight="bold", color="white")

    chart_path = output_dir / f"{seq_name}_xp_chart.png"
    plt.savefig(str(chart_path), dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✓ Chart saved → {chart_path}")


# ---------------------------------------------------------------------------
# Per-sequence processor
# ---------------------------------------------------------------------------

def process_sequence(seq_dir: Path, output_dir: Path):
    name = seq_dir.name
    print(f"\n{'='*60}")
    print(f"  Processing  {name}")
    print(f"{'='*60}")

    gt_path    = seq_dir / "gt"  / "gt.txt"
    img_dir    = seq_dir / "img1"
    seqinfo_p  = seq_dir / "seqinfo.ini"
    gameinfo_p = seq_dir / "gameinfo.ini"

    if not gt_path.exists() or not img_dir.exists():
        print("  [SKIP] missing gt or img1/")
        return

    seqinfo = configparser.ConfigParser()
    seqinfo.read(seqinfo_p)
    fps     = float(seqinfo["Sequence"].get("framerate", 25))
    img_ext = seqinfo["Sequence"].get("imext", ".jpg")

    team_map, ball_id, jersey_map = parse_gameinfo(gameinfo_p)
    frame_data = parse_gt(gt_path)
    frames     = sorted(frame_data.keys())
    if not frames:
        print("  [SKIP] empty gt")
        return

    n_pl = sum(1 for v in team_map.values() if v in ("left","right"))
    print(f"  Frames: {len(frames)}  |  FPS: {fps}  |  "
          f"Players: {n_pl}  |  Ball track_id: {ball_id}")

    # ── Pre-detect pass events and compute xP for cumulative tracking ──
    xp_tm, xp_bid = _xp_parse_gameinfo(gameinfo_p)
    pass_events   = _xp_detect_passes(frame_data, xp_tm, xp_bid, name)
    # Build lookup: end-frame → [(passer_tid, xP)]
    pass_at_frame: dict[int, list[tuple]] = defaultdict(list)
    for p in pass_events:
        if p["outcome"] == 1:      # only credit completed passes
            pbbox = frame_data.get(p["frame_start"], {}).get(p["passer_tid"])
            rbbox = frame_data.get(p["frame_end"],   {}).get(p["receiver_tid"])
            bbbox = frame_data.get(p["frame_start"], {}).get(xp_bid) if xp_bid else None
            if pbbox and rbbox:
                xp = compute_xp(pbbox, rbbox, bbbox, xp_tm.get(p["passer_tid"]))
                pass_at_frame[p["frame_end"]].append((p["passer_tid"], xp))

    # Running cumulative xP per player (used for overlay + chart)
    running_xp: dict[int, float]         = defaultdict(float)
    # History for chart: {tid: [(frame, cumxp)]}
    cumxp_history: dict[int, list]       = defaultdict(list)

    # First-image size
    first_img = img_dir / f"{frames[0]:06d}{img_ext}"
    if not first_img.exists():
        imgs = sorted(img_dir.glob(f"*{img_ext}"))
        first_img = imgs[0] if imgs else None
    if first_img is None:
        print("  [SKIP] no images")
        return

    sample = cv2.imread(str(first_img))
    if sample is None:
        print("  [SKIP] unreadable image")
        return
    H, W = sample.shape[:2]

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{name}_passing.mp4"
    writer   = cv2.VideoWriter(str(out_path),
                               cv2.VideoWriter_fourcc(*"mp4v"),
                               fps, (W, H))

    total = len(frames)
    for i, fid in enumerate(frames):
        if (i+1) % 100 == 0 or i == 0:
            print(f"  Frame {i+1}/{total} …")

        # Update cumulative xP when a pass lands this frame
        for (passer_tid, xp_val) in pass_at_frame.get(fid, []):
            running_xp[passer_tid] += xp_val
            cumxp_history[passer_tid].append((fid, running_xp[passer_tid]))

        img_path = img_dir / f"{fid:06d}{img_ext}"
        if not img_path.exists():
            continue
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue

        draw_player_overlays(frame, fid, frame_data, team_map, jersey_map,
                             ball_id, running_xp)
        writer.write(frame)

    writer.release()
    print(f"  ✓ MP4 saved → {out_path}")

    # ── Generate cumulative-xP chart ──
    generate_xp_chart(cumxp_history, jersey_map, team_map,
                      name, output_dir, frames[-1] if frames else 0)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Passing Polygon CV — Ball Carrier Mode")
    parser.add_argument("--seq", nargs="*", metavar="SNMOT-XXX")
    args = parser.parse_args()

    if not TRAIN_DIR.exists():
        sys.exit(f"[ERROR] train/ not found: {TRAIN_DIR}")

    all_seqs = sorted(d for d in TRAIN_DIR.iterdir()
                      if d.is_dir() and d.name.startswith("SNMOT"))
    if not all_seqs:
        sys.exit("[ERROR] No sequences found")

    selected = ([TRAIN_DIR / s for s in args.seq] if args.seq else all_seqs)
    for s in selected:
        if not s.exists():
            sys.exit(f"[ERROR] Not found: {s}")

    print(f"Passing Polygon CV  —  Ball-Carrier Mode")
    print(f"Sequences : {len(selected)}   Output : {OUTPUT_DIR}")

    # Train global xP model once before processing any sequence
    build_xp_model()

    for seq_dir in selected:
        process_sequence(seq_dir, OUTPUT_DIR)

    print(f"\n✅  Done. MP4s + charts → {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()

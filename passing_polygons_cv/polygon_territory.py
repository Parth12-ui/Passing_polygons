"""
Territorial Polygon Formation + xG Analysis
============================================
Standalone script — does NOT modify main.py or xp_model.py.

For every SNMOT-* sequence in train/:
  1. Parse gameinfo.ini → team_map, ball_id, jersey_map
  2. Parse gt/gt.txt    → per-frame bounding boxes
  3. Build per-team convex-hull territory polygons per frame
       - Finds the largest proximity-connected cluster of players
         (players beyond MAX_PLAYER_DIST_PX from all teammates are excluded)
       - Takes convex hull of that cluster — no opposition-blocking check
  4. Compute contested area (home ∩ away) via cv2.intersectConvexConvex
  5. Classify "attacking frame" when ball crosses opposition half
  6. Extract xG features and score with a global LogisticRegression model
     trained on geometric danger-zone heuristics (no external annotations)
  7. Render annotated video:
       - Blue  fill  = home (left) territory
       - Red   fill  = away (right) territory
       - Orange fill = contested overlap
       - Danger ring around ball, scaling yellow→red with xG
       - HUD panel with live area % + xG
  8. Write per-frame CSV with area stats + xG
  9. Generate per-sequence chart:
       - Scatter: defending polygon area % vs xG
       - Time series: home/away area % + xG overlaid

Outputs (per sequence):
    output/<seq>_territory.mp4
    output/<seq>_territory.csv
    output/<seq>_territory_xg_chart.png

Usage:
    python polygon_territory.py                     # all sequences
    python polygon_territory.py --seq SNMOT-170     # single sequence
"""

import argparse
import configparser
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
TRAIN_DIR  = Path(__file__).parent / "train"
OUTPUT_DIR = Path(__file__).parent / "output"

# ---------------------------------------------------------------------------
# Visual constants (BGR)
# ---------------------------------------------------------------------------
HOME_FILL_COLOUR       = (180,  60,  20)   # blue-ish fill  – home / left team
AWAY_FILL_COLOUR       = ( 20,  50, 180)   # red-ish fill   – away / right team
CONTESTED_FILL_COLOUR  = (  0, 140, 255)   # orange fill    – contested overlap

HOME_EDGE_COLOUR       = (255, 160,  80)   # bright blue outline
AWAY_EDGE_COLOUR       = ( 80, 100, 255)   # bright red outline

HOME_BOX_COLOUR        = (210, 140,  30)   # amber  – player bounding box
AWAY_BOX_COLOUR        = ( 30,  30, 220)   # red    – player bounding box
HOME_NODE_COLOUR       = (255, 220, 100)
AWAY_NODE_COLOUR       = (120, 120, 255)
BALL_COLOUR            = ( 50, 230, 255)   # bright yellow

TERRITORY_FILL_ALPHA   = 0.28   # alpha for home/away fills
CONTESTED_FILL_ALPHA   = 0.45   # alpha for contested fill (drawn on top)

NODE_RADIUS            = 5
BALL_RADIUS            = 6

# Max pixel distance between two teammates for them to be considered
# in the same formation cluster.  Players farther than this from every
# other teammate are excluded from the polygon (e.g. lone goalkeeper).
# ≈ 16 m on a 1920-wide image of a standard 105 m pitch.
MAX_PLAYER_DIST_PX     = 300

# xG danger zone: within this fraction of pitch width from the opposition goal
DANGER_ZONE_X_FRAC     = 0.25

# Fewer than this many defenders in the corridor → xG label = 1
DANGER_MAX_DEFENDERS   = 2

# Half-width of defender corridor as fraction of image height
DEFENDER_CORRIDOR_W    = 0.15

# ---------------------------------------------------------------------------
# CSV / Feature columns
# ---------------------------------------------------------------------------
TERRITORY_CSV_FIELDS = [
    "frame", "attacking_team",
    "home_n_clusters", "away_n_clusters",
    "home_area_px", "away_area_px", "contested_area_px", "pitch_area_px",
    "home_pct", "away_pct", "contested_pct",
    "xG", "ball_x_norm", "n_attackers_in_opp_half",
    "n_defenders_between_ball_goal", "def_polygon_area_pct",
]

XG_FEATURE_COLS = [
    "ball_dist_to_goal_norm",
    "ball_angle_to_goal",
    "n_attackers_in_opp_half",
    "n_defenders_between_ball_goal",
    "atk_polygon_area_pct",
    "def_polygon_area_pct",
    "contested_area_pct",
    "ball_x_norm",
    "ball_y_norm",
]

# ---------------------------------------------------------------------------
# Global xG model state (trained once before any sequence is processed)
# ---------------------------------------------------------------------------
_XG_MODEL:    LogisticRegression | None = None
_XG_SCALER:   StandardScaler     | None = None
_XG_CLS1_IDX: int = 1

# ---------------------------------------------------------------------------
# Parsing helpers  (independent of main.py / xp_model.py)
# ---------------------------------------------------------------------------

def parse_gameinfo(path: Path):
    """Return team_map {tid→'left'|'right'|None}, ball_id, jersey_map."""
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
        value  = value.strip()
        role   = value.split(";")[0].strip().lower()
        parts  = value.split(";")
        jersey_map[idx] = parts[1].strip() if len(parts) > 1 else "?"
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
    return team_map, ball_id, jersey_map


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
            fid = int(p[0])
            tid = int(p[1])
            fd[fid][tid] = (float(p[2]), float(p[3]), float(p[4]), float(p[5]))
    return fd


def read_seq_dims(seq_dir: Path):
    """Read (width, height) from seqinfo.ini; fallback 1920×1080."""
    sinfo = configparser.ConfigParser()
    sinfo.read(seq_dir / "seqinfo.ini")
    try:
        w = int(sinfo["Sequence"].get("imwidth",  1920))
        h = int(sinfo["Sequence"].get("imheight", 1080))
        return w, h
    except Exception:
        return 1920, 1080

# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def foot_point(x, y, w, h):
    return (x + w / 2.0, y + h)


def centre_point(x, y, w, h):
    return (x + w / 2.0, y + h / 2.0)


def dist2(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def segments_intersect(p1, p2, p3, p4):
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    d1 = cross(p3, p4, p1)
    d2 = cross(p3, p4, p2)
    d3 = cross(p1, p2, p3)
    d4 = cross(p1, p2, p4)
    return (((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and
            ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)))


def bbox_edges(x, y, w, h, expand=0):
    x1, y1 = x - expand, y - expand
    x2, y2 = x + w + expand, y + h + expand
    return [
        ((x1, y1), (x2, y1)),
        ((x2, y1), (x2, y2)),
        ((x2, y2), (x1, y2)),
        ((x1, y2), (x1, y1)),
    ]


def clamp_pt(pt, W, H):
    return (int(max(0, min(W - 1, pt[0]))), int(max(0, min(H - 1, pt[1]))))

# ---------------------------------------------------------------------------
# Polygon builder  (proximity-cluster convex hull)
# ---------------------------------------------------------------------------

def build_team_clusters(player_bboxes: list):
    """
    Build convex-hull polygons for ALL proximity-connected clusters of one team.

    Strategy
    --------
    1. Compute foot-points for every player in this team.
    2. Build a proximity graph: two players are adjacent iff they are
       within MAX_PLAYER_DIST_PX of each other.
    3. BFS to find every connected component.
    4. Build a convex hull for each component with ≥ 3 players.
    5. Return them sorted by area (largest first).

    This means a forward line separated from the rest, or a lone striker
    with a partner, will produce its own smaller polygon instead of being
    absorbed into (or excluded from) the main hull.

    Parameters
    ----------
    player_bboxes : list of (x, y, w, h)

    Returns
    -------
    list of numpy int32 arrays (N, 1, 2), one per valid cluster.  Empty list
    if no cluster has ≥ 3 players.
    """
    if len(player_bboxes) < 3:
        return []

    pts = [foot_point(*b) for b in player_bboxes]
    n   = len(pts)

    # ── Build proximity graph ────────────────────────────────────────────
    adj = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = math.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1])
            if d <= MAX_PLAYER_DIST_PX:
                adj[i].append(j)
                adj[j].append(i)

    # ── BFS: collect ALL connected components ────────────────────────────
    visited    = [False] * n
    components = []
    for start in range(n):
        if visited[start]:
            continue
        comp  = []
        queue = [start]
        visited[start] = True
        while queue:
            node = queue.pop(0)
            comp.append(node)
            for nb in adj[node]:
                if not visited[nb]:
                    visited[nb] = True
                    queue.append(nb)
        components.append(comp)

    # ── Convex hull per valid cluster ────────────────────────────────────
    hulls = []
    for comp in components:
        if len(comp) < 3:
            continue
        cluster_pts = np.array([pts[i] for i in comp], dtype=np.float32)
        hull = cv2.convexHull(cluster_pts.reshape(-1, 1, 2))
        hulls.append(hull.astype(np.int32))

    # Largest cluster first
    hulls.sort(key=lambda h: cv2.contourArea(h), reverse=True)
    return hulls

# ---------------------------------------------------------------------------
# Polygon intersection
# ---------------------------------------------------------------------------

def polygon_intersection(hull_a, hull_b):
    """
    Returns (area, intersection_polygon | None).
    hull_a, hull_b must be int32 numpy arrays of shape (N, 1, 2).
    """
    if hull_a is None or hull_b is None:
        return 0.0, None
    if len(hull_a) < 3 or len(hull_b) < 3:
        return 0.0, None
    try:
        ret, inter = cv2.intersectConvexConvex(
            hull_a.astype(np.float32),
            hull_b.astype(np.float32),
        )
        if ret > 0 and inter is not None and len(inter) >= 3:
            area = float(cv2.contourArea(inter))
            return area, inter.astype(np.int32)
    except Exception:
        pass
    return 0.0, None

# ---------------------------------------------------------------------------
# xG feature extraction
# ---------------------------------------------------------------------------

def count_defenders_between_ball_goal(ball_pos, def_bboxes, atk_side, W, H):
    """Count opposition defenders in the corridor between ball and goal."""
    bx, by   = ball_pos
    goal_x   = float(W) if atk_side == "left" else 0.0
    goal_y   = H / 2.0
    count    = 0
    for (dx, dy, dw, dh) in def_bboxes:
        dfx = dx + dw / 2.0
        dfy = dy + dh
        # x-range check: defender must be between ball and goal
        if atk_side == "left":
            in_x = bx < dfx < goal_x
        else:
            in_x = goal_x < dfx < bx
        if not in_x:
            continue
        # Corridor centre at this x via linear interpolation
        span = goal_x - bx
        cy   = by + ((dfx - bx) / span) * (goal_y - by) if abs(span) > 1 else by
        if abs(dfy - cy) < H * DEFENDER_CORRIDOR_W:
            count += 1
    return count


def extract_xg_features(ball_pos, atk_side, atk_bboxes, def_bboxes,
                         home_area, away_area, contested_area, W, H):
    """Return feature dict keyed by XG_FEATURE_COLS."""
    bx, by = ball_pos
    goal_x = float(W) if atk_side == "left" else 0.0
    goal_y = H / 2.0

    dist_to_goal      = math.hypot(bx - goal_x, by - goal_y)
    max_dist          = math.hypot(W, H)
    dist_norm         = dist_to_goal / max_dist if max_dist > 0 else 0.0

    angle_to_goal     = abs(math.degrees(math.atan2(goal_y - by, goal_x - bx)))

    ball_x_norm       = (bx / W) if atk_side == "left" else (1.0 - bx / W)
    ball_y_norm       = abs(by - H / 2.0) / (H / 2.0) if H > 0 else 0.0

    mid_x             = W / 2.0
    n_atk_opp         = sum(
        1 for (ax, ay, aw, ah) in atk_bboxes
        if (ax + aw / 2.0 > mid_x if atk_side == "left" else ax + aw / 2.0 < mid_x)
    )
    n_def_between     = count_defenders_between_ball_goal(ball_pos, def_bboxes, atk_side, W, H)

    pitch_area        = float(W * H) or 1.0
    atk_area          = home_area if atk_side == "left" else away_area
    def_area          = away_area if atk_side == "left" else home_area

    return {
        "ball_dist_to_goal_norm":        round(dist_norm, 4),
        "ball_angle_to_goal":            round(angle_to_goal, 2),
        "n_attackers_in_opp_half":       n_atk_opp,
        "n_defenders_between_ball_goal": n_def_between,
        "atk_polygon_area_pct":          round(atk_area / pitch_area * 100, 4),
        "def_polygon_area_pct":          round(def_area / pitch_area * 100, 4),
        "contested_area_pct":            round(contested_area / pitch_area * 100, 4),
        "ball_x_norm":                   round(ball_x_norm, 4),
        "ball_y_norm":                   round(ball_y_norm, 4),
    }


def make_xg_label(ball_pos, atk_side, def_bboxes, W, H):
    """
    Geometric danger-zone label (no external annotations required).
    1 = ball in danger zone with few defenders between ball and goal.
    0 = safe / no clear shooting opportunity.
    """
    bx = ball_pos[0]
    if atk_side == "left":
        in_danger = bx > W * (1.0 - DANGER_ZONE_X_FRAC)
    else:
        in_danger = bx < W * DANGER_ZONE_X_FRAC
    if not in_danger:
        return 0
    n_def = count_defenders_between_ball_goal(ball_pos, def_bboxes, atk_side, W, H)
    return 1 if n_def < DANGER_MAX_DEFENDERS else 0

# ---------------------------------------------------------------------------
# Global xG model  (trained once across all sequences before video pass)
# ---------------------------------------------------------------------------

def build_xg_model(W=1920, H=1080):
    """Train global Logistic Regression xG model and store in module globals."""
    global _XG_MODEL, _XG_SCALER, _XG_CLS1_IDX
    print("\n[xG] Building global xG model …")

    samples, labels = [], []

    for seq_dir in sorted(TRAIN_DIR.iterdir()):
        if not seq_dir.is_dir() or not seq_dir.name.startswith("SNMOT"):
            continue
        gt_path = seq_dir / "gt" / "gt.txt"
        gi_path = seq_dir / "gameinfo.ini"
        if not gt_path.exists():
            continue

        seq_W, seq_H  = read_seq_dims(seq_dir)
        team_map, ball_id, _ = parse_gameinfo(gi_path)
        frame_data            = parse_gt(gt_path)
        mid_x                 = seq_W / 2.0

        for fid, cur in frame_data.items():
            ball_bbox = cur.get(ball_id) if ball_id else None
            if ball_bbox is None:
                continue
            ball_pos  = centre_point(*ball_bbox)
            bx        = ball_pos[0]

            left_bb  = [b for t, b in cur.items() if team_map.get(t) == "left"]
            right_bb = [b for t, b in cur.items() if team_map.get(t) == "right"]

            # Determine attacking side from ball position
            if left_bb  and bx > mid_x:
                atk_side = "left"
            elif right_bb and bx < mid_x:
                atk_side = "right"
            else:
                continue

            atk_bb = left_bb  if atk_side == "left"  else right_bb
            def_bb = right_bb if atk_side == "left"  else left_bb

            # Quick (non-pruned) hull areas for training features only
            def quick_area(blist):
                if len(blist) < 3:
                    return 0.0
                pts = np.array([foot_point(*b) for b in blist], dtype=np.float32)
                return float(cv2.contourArea(cv2.convexHull(pts.reshape(-1, 1, 2))))

            home_area = quick_area(left_bb)
            away_area = quick_area(right_bb)

            feats = extract_xg_features(
                ball_pos, atk_side, atk_bb, def_bb,
                home_area, away_area, 0.0, seq_W, seq_H,
            )
            label = make_xg_label(ball_pos, atk_side, def_bb, seq_W, seq_H)
            samples.append([feats[c] for c in XG_FEATURE_COLS])
            labels.append(label)

    if len(samples) < 10:
        print("[xG] Insufficient data — xG will default to 0.0")
        return

    X = np.array(samples, dtype=float)
    y = np.array(labels,  dtype=int)
    n_pos = int(y.sum())
    print(f"[xG] Training samples: {len(y)}  (danger={n_pos}, safe={len(y)-n_pos})")

    if len(set(y)) < 2:
        print("[xG] Only one class present — model skipped")
        return

    scaler = StandardScaler()
    X_s    = scaler.fit_transform(X)
    model  = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_s, y)

    _XG_MODEL    = model
    _XG_SCALER   = scaler
    _XG_CLS1_IDX = list(model.classes_).index(1)
    print(f"[xG] Model ready  (train accuracy: {100*model.score(X_s, y):.1f}%)")


def compute_xg(feats: dict) -> float:
    """Return predicted xG ∈ [0, 1] for one attacking frame."""
    if _XG_MODEL is None or _XG_SCALER is None:
        return 0.0
    X   = np.array([[feats[c] for c in XG_FEATURE_COLS]], dtype=float)
    X_s = _XG_SCALER.transform(X)
    return round(float(_XG_MODEL.predict_proba(X_s)[0][_XG_CLS1_IDX]), 4)

# ---------------------------------------------------------------------------
# Frame renderer
# ---------------------------------------------------------------------------

def draw_player_boxes(frame_img, players, box_col, node_col, jersey_map, W, H):
    """Draw bounding boxes, foot nodes, and jersey labels."""
    for tid, (x, y, w, h) in players.items():
        x1 = max(0, int(x));       y1 = max(0, int(y))
        x2 = min(W - 1, int(x+w)); y2 = min(H - 1, int(y+h))
        cv2.rectangle(frame_img, (x1, y1), (x2, y2), box_col, 2)
        fp = clamp_pt(foot_point(x, y, w, h), W, H)
        cv2.circle(frame_img, fp, NODE_RADIUS + 2, (0, 0, 0),  -1)
        cv2.circle(frame_img, fp, NODE_RADIUS,     node_col,   -1)
        jersey = jersey_map.get(tid, "?")
        label  = f"#{jersey}"
        fs, th = 0.42, 1
        (tw, th_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
        lx, ly = x1, max(th_h + 2, y1 - 2)
        cv2.rectangle(frame_img, (lx, ly-th_h-2), (lx+tw+4, ly), box_col, -1)
        cv2.putText(frame_img, label, (lx+2, ly-1),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), th, cv2.LINE_AA)


def draw_territory_overlay(frame_img, home_hulls, away_hulls, inter_polys):
    """
    Render semi-transparent territory fills for ALL clusters then outlines.

    Parameters
    ----------
    home_hulls  : list of int32 hull arrays for home team clusters
    away_hulls  : list of int32 hull arrays for away team clusters
    inter_polys : list of int32 polygon arrays (all pairwise intersections)
    """
    # Layer 1: all home + away cluster fills
    overlay1 = frame_img.copy()
    for hull in home_hulls:
        if len(hull) >= 3:
            cv2.fillPoly(overlay1, [hull], HOME_FILL_COLOUR)
    for hull in away_hulls:
        if len(hull) >= 3:
            cv2.fillPoly(overlay1, [hull], AWAY_FILL_COLOUR)
    cv2.addWeighted(overlay1, TERRITORY_FILL_ALPHA,
                    frame_img, 1.0 - TERRITORY_FILL_ALPHA, 0, frame_img)

    # Layer 2: all contested fills (stronger alpha, drawn on top)
    if inter_polys:
        overlay2 = frame_img.copy()
        for poly in inter_polys:
            if len(poly) >= 3:
                cv2.fillPoly(overlay2, [poly], CONTESTED_FILL_COLOUR)
        cv2.addWeighted(overlay2, CONTESTED_FILL_ALPHA,
                        frame_img, 1.0 - CONTESTED_FILL_ALPHA, 0, frame_img)

    # Outlines for every cluster (full opacity)
    for hull in home_hulls:
        if len(hull) >= 3:
            cv2.polylines(frame_img, [hull], True, HOME_EDGE_COLOUR, 2, cv2.LINE_AA)
    for hull in away_hulls:
        if len(hull) >= 3:
            cv2.polylines(frame_img, [hull], True, AWAY_EDGE_COLOUR, 2, cv2.LINE_AA)
    for poly in inter_polys:
        if len(poly) >= 3:
            cv2.polylines(frame_img, [poly], True, CONTESTED_FILL_COLOUR, 1, cv2.LINE_AA)


def draw_midfield_line(frame_img, W, H):
    """Dashed vertical midfield line to visualise opposition halves."""
    x     = W // 2
    dash  = 12
    gap   = 8
    y     = 0
    col   = (180, 180, 180)
    while y < H:
        cv2.line(frame_img, (x, y), (x, min(H-1, y + dash)), col, 1, cv2.LINE_AA)
        y += dash + gap


def draw_ball_danger_ring(frame_img, ball_bp, xg_val):
    """Scale + hue shift ring around ball; yellow (low xG) → red (high xG)."""
    if xg_val <= 0.05:
        return
    radius = int(15 + xg_val * 30)
    g      = int(255 * (1.0 - xg_val))   # full red at xG=1, yellow at xG=0
    colour = (0, g, 255)                  # BGR
    cv2.circle(frame_img, ball_bp, radius + 2, (0, 0, 0), 2)
    cv2.circle(frame_img, ball_bp, radius,     colour,    2, cv2.LINE_AA)


def draw_hud(frame_img, home_pct, away_pct, contested_pct, xg_val, atk_side, W, H):
    """Stats panel — top-right corner."""
    atk_label = {"left": "HOME", "right": "AWAY"}.get(atk_side, "—")
    lines = [
        (f"HOME area:    {home_pct:5.1f}%",      HOME_EDGE_COLOUR),
        (f"AWAY area:    {away_pct:5.1f}%",      AWAY_EDGE_COLOUR),
        (f"CONTESTED:    {contested_pct:5.1f}%", CONTESTED_FILL_COLOUR),
        (f"Attacking:    {atk_label}",            (200, 200, 200)),
        (f"xG  :         {xg_val:.3f}",          (100, 100, 255) if xg_val < 0.3
                                                  else (0, 180, 255) if xg_val < 0.6
                                                  else (0, 80, 255)),
    ]
    fs, th_l = 0.40, 1
    panel_w  = 215
    panel_h  = len(lines) * 20 + 12
    px, py   = W - panel_w - 10, 10

    cv2.rectangle(frame_img, (px-5, py-5), (px+panel_w, py+panel_h), (0, 0, 0),   -1)
    cv2.rectangle(frame_img, (px-5, py-5), (px+panel_w, py+panel_h), (80,80,80),   1)

    for i, (text, col) in enumerate(lines):
        ly = py + 16 + i * 20
        cv2.putText(frame_img, text, (px, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, col, th_l, cv2.LINE_AA)


def draw_legend(frame_img, H):
    """Colour-coded legend in the bottom-left corner."""
    items = [
        ("Home territory",  HOME_EDGE_COLOUR),
        ("Away territory",  AWAY_EDGE_COLOUR),
        ("Contested area",  CONTESTED_FILL_COLOUR),
        ("xG danger ring",  (0, 100, 255)),
    ]
    ly = H - 12 - len(items) * 18
    for text, col in items:
        cv2.rectangle(frame_img, (8, ly-10), (24, ly+4), col, -1)
        cv2.putText(frame_img, text, (30, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
        ly += 18

# ---------------------------------------------------------------------------
# Per-frame processing  (geometry + xG + rendering)
# ---------------------------------------------------------------------------

def process_frame(frame_img, fid, frame_data, team_map, jersey_map, ball_id, W, H):
    """
    Build polygons, compute areas, predict xG, render all overlays.
    Returns a stats dict (one CSV row).
    """
    cur       = frame_data.get(fid, {})
    left_pl   = {t: b for t, b in cur.items() if team_map.get(t) == "left"}
    right_pl  = {t: b for t, b in cur.items() if team_map.get(t) == "right"}
    left_bb   = list(left_pl.values())
    right_bb  = list(right_pl.values())

    # ── Territory clusters (ALL proximity-connected groups per team) ──────
    home_hulls = build_team_clusters(left_bb)
    away_hulls = build_team_clusters(right_bb)

    home_area      = sum(float(cv2.contourArea(h)) for h in home_hulls)
    away_area      = sum(float(cv2.contourArea(h)) for h in away_hulls)

    # Contested = sum of all pairwise cluster intersections
    contested_area = 0.0
    inter_polys    = []
    for hh in home_hulls:
        for ah in away_hulls:
            a, poly = polygon_intersection(hh, ah)
            if a > 0 and poly is not None:
                contested_area += a
                inter_polys.append(poly)

    pitch_area    = float(W * H)
    home_pct      = home_area      / pitch_area * 100 if pitch_area else 0.0
    away_pct      = away_area      / pitch_area * 100 if pitch_area else 0.0
    contested_pct = contested_area / pitch_area * 100 if pitch_area else 0.0

    # ── Ball / attacking side / xG ────────────────────────────────────────
    ball_bbox     = cur.get(ball_id) if ball_id else None
    ball_pos      = None
    ball_bp       = None
    atk_side      = None
    xg_val        = 0.0
    ball_x_norm   = 0.0
    n_atk_opp     = 0
    n_def_between = 0
    def_pct       = 0.0

    if ball_bbox is not None:
        ball_pos = centre_point(*ball_bbox)
        ball_bp  = clamp_pt(ball_pos, W, H)
        bx       = ball_pos[0]
        mid_x    = W / 2.0

        if left_bb  and bx > mid_x:
            atk_side = "left"
        elif right_bb and bx < mid_x:
            atk_side = "right"

        if atk_side is not None:
            atk_bb = left_bb  if atk_side == "left"  else right_bb
            def_bb = right_bb if atk_side == "left"  else left_bb

            feats         = extract_xg_features(
                ball_pos, atk_side, atk_bb, def_bb,
                home_area, away_area, contested_area, W, H,
            )
            xg_val        = compute_xg(feats)
            ball_x_norm   = feats["ball_x_norm"]
            n_atk_opp     = feats["n_attackers_in_opp_half"]
            n_def_between = feats["n_defenders_between_ball_goal"]
            def_pct       = feats["def_polygon_area_pct"]

    # ── Draw ──────────────────────────────────────────────────────────────
    draw_midfield_line(frame_img, W, H)
    draw_territory_overlay(frame_img, home_hulls, away_hulls, inter_polys)
    draw_player_boxes(frame_img, left_pl,  HOME_BOX_COLOUR, HOME_NODE_COLOUR, jersey_map, W, H)
    draw_player_boxes(frame_img, right_pl, AWAY_BOX_COLOUR, AWAY_NODE_COLOUR, jersey_map, W, H)

    if ball_bp is not None:
        draw_ball_danger_ring(frame_img, ball_bp, xg_val)
        cv2.circle(frame_img, ball_bp, BALL_RADIUS + 2, (0, 0, 0),  -1)
        cv2.circle(frame_img, ball_bp, BALL_RADIUS,     BALL_COLOUR, -1)

    draw_hud(frame_img, home_pct, away_pct, contested_pct, xg_val, atk_side, W, H)
    draw_legend(frame_img, H)

    return {
        "frame":                         fid,
        "attacking_team":                atk_side or "none",
        "home_n_clusters":               len(home_hulls),
        "away_n_clusters":               len(away_hulls),
        "home_area_px":                  round(home_area),
        "away_area_px":                  round(away_area),
        "contested_area_px":             round(contested_area),
        "pitch_area_px":                 round(pitch_area),
        "home_pct":                      round(home_pct, 2),
        "away_pct":                      round(away_pct, 2),
        "contested_pct":                 round(contested_pct, 2),
        "xG":                            xg_val,
        "ball_x_norm":                   round(ball_x_norm, 4),
        "n_attackers_in_opp_half":       n_atk_opp,
        "n_defenders_between_ball_goal": n_def_between,
        "def_polygon_area_pct":          round(def_pct, 2),
    }

# ---------------------------------------------------------------------------
# Chart generator
# ---------------------------------------------------------------------------

def generate_territory_chart(records, seq_name, output_dir):
    """
    Two-panel figure:
      Left  — Scatter: defending polygon area % vs xG  (with trend line)
      Right — Time series: home/away area % over frames with xG on twin axis
    """
    if not records:
        return

    BG_DARK  = "#1a1a1a"
    BG_PANEL = "#111111"

    df_frames = [r["frame"]         for r in records]
    df_home   = [r["home_pct"]      for r in records]
    df_away   = [r["away_pct"]      for r in records]
    df_xg     = [r["xG"]            for r in records]

    # Attacking frames only (xG > 0)
    atk_recs = [
        (r["def_polygon_area_pct"], r["xG"], r["attacking_team"])
        for r in records if r["xG"] > 0 and r["attacking_team"] != "none"
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5),
                                   gridspec_kw={"wspace": 0.38})
    fig.set_facecolor(BG_DARK)
    fig.suptitle(f"Territorial Analysis — {seq_name}",
                 fontsize=13, fontweight="bold", color="white")

    # ── Left: Scatter ─────────────────────────────────────────────────────
    ax1.set_facecolor(BG_PANEL)
    if atk_recs:
        home_pts = [(d, x) for d, x, t in atk_recs if t == "left"]
        away_pts = [(d, x) for d, x, t in atk_recs if t == "right"]
        if home_pts:
            d_, x_ = zip(*home_pts)
            ax1.scatter(d_, x_, c="#FF7744", alpha=0.50, s=14,
                        label="Home attacking", zorder=3)
        if away_pts:
            d_, x_ = zip(*away_pts)
            ax1.scatter(d_, x_, c="#4477FF", alpha=0.50, s=14,
                        label="Away attacking", zorder=3)

        # Smoothed trend across all attacking frames
        all_d = sorted(r[0] for r in atk_recs)
        all_x = [r[1] for r in sorted(atk_recs, key=lambda r: r[0])]
        if len(all_d) > 15:
            window   = max(1, len(all_d) // 20)
            smooth_x = np.convolve(all_x, np.ones(window) / window, mode="valid")
            smooth_d = all_d[:len(smooth_x)]
            ax1.plot(smooth_d, smooth_x, color="white", linewidth=1.6,
                     alpha=0.75, label="Trend", zorder=4)

    ax1.set_xlabel("Defending Polygon Area (%)", fontsize=9, color="white")
    ax1.set_ylabel("xG Score", fontsize=9, color="white")
    ax1.set_title("Defending Compactness vs Attack Danger", fontsize=9, color="white")
    ax1.legend(fontsize=7, loc="upper right",
               facecolor="#2a2a2a", labelcolor="white", framealpha=0.75)
    ax1.grid(True, alpha=0.20)

    # ── Right: Time series ────────────────────────────────────────────────
    ax2.set_facecolor(BG_PANEL)
    ax2.plot(df_frames, df_home, color="#FF9944", linewidth=1.2,
             alpha=0.85, label="Home area %")
    ax2.plot(df_frames, df_away, color="#4466FF", linewidth=1.2,
             alpha=0.85, label="Away area %")
    ax2.set_xlabel("Frame", fontsize=9, color="white")
    ax2.set_ylabel("Territory Area (%)", fontsize=9, color="white")
    ax2.set_title("Area Over Time + xG", fontsize=9, color="white")

    ax2r = ax2.twinx()
    ax2r.plot(df_frames, df_xg, color="#FF3333", linewidth=0.9,
              alpha=0.65, label="xG")
    ax2r.set_ylabel("xG", fontsize=9, color="#FF6666")
    ax2r.tick_params(colors="#FF6666", labelsize=8)
    for spine in ax2r.spines.values():
        spine.set_edgecolor("#444444")

    lines1, lbl1 = ax2.get_legend_handles_labels()
    lines2, lbl2 = ax2r.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, lbl1 + lbl2, fontsize=7, loc="upper left",
               facecolor="#2a2a2a", labelcolor="white", framealpha=0.75)
    ax2.grid(True, alpha=0.20)

    for ax in (ax1, ax2):
        ax.tick_params(colors="white", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#444444")

    chart_path = output_dir / f"{seq_name}_territory_xg_chart.png"
    plt.savefig(str(chart_path), dpi=120, bbox_inches="tight", facecolor=BG_DARK)
    plt.close(fig)
    print(f"  ✓ Chart saved    → {chart_path}")

# ---------------------------------------------------------------------------
# CSV writer
# ---------------------------------------------------------------------------

def write_territory_csv(records, output_dir, seq_name):
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{seq_name}_territory.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TERRITORY_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(records)
    return out_path

# ---------------------------------------------------------------------------
# Per-sequence processor
# ---------------------------------------------------------------------------

def process_sequence(seq_dir: Path, output_dir: Path):
    name = seq_dir.name
    print(f"\n{'='*60}")
    print(f"  Territory — {name}")
    print(f"{'='*60}")

    gt_path    = seq_dir / "gt"    / "gt.txt"
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

    n_pl = sum(1 for v in team_map.values() if v in ("left", "right"))
    print(f"  Frames: {len(frames)}  |  FPS: {fps}  |  Players: {n_pl}  |  Ball: {ball_id}")

    # First image → determine W, H
    first_img = img_dir / f"{frames[0]:06d}{img_ext}"
    if not first_img.exists():
        imgs      = sorted(img_dir.glob(f"*{img_ext}"))
        first_img = imgs[0] if imgs else None
    if first_img is None:
        print("  [SKIP] no images found")
        return

    sample = cv2.imread(str(first_img))
    if sample is None:
        print("  [SKIP] unreadable image")
        return
    H, W = sample.shape[:2]

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{name}_territory.mp4"
    writer   = cv2.VideoWriter(str(out_path),
                               cv2.VideoWriter_fourcc(*"mp4v"),
                               fps, (W, H))
    records = []
    total   = len(frames)

    for i, fid in enumerate(frames):
        if (i + 1) % 100 == 0 or i == 0:
            print(f"  Frame {i+1}/{total} …")

        img_path = img_dir / f"{fid:06d}{img_ext}"
        if not img_path.exists():
            continue
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue

        row = process_frame(frame, fid, frame_data, team_map, jersey_map, ball_id, W, H)
        records.append(row)
        writer.write(frame)

    writer.release()
    print(f"  ✓ MP4 saved      → {out_path}")

    csv_path = write_territory_csv(records, output_dir, name)
    print(f"  ✓ CSV saved      → {csv_path}")

    generate_territory_chart(records, name, output_dir)

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Territorial Polygon Formation + xG Analysis"
    )
    parser.add_argument("--seq", nargs="*", metavar="SNMOT-XXX",
                        help="Specific sequence(s) to process (default: all)")
    args = parser.parse_args()

    if not TRAIN_DIR.exists():
        sys.exit(f"[ERROR] train/ not found: {TRAIN_DIR}")

    all_seqs = sorted(d for d in TRAIN_DIR.iterdir()
                      if d.is_dir() and d.name.startswith("SNMOT"))
    if not all_seqs:
        sys.exit("[ERROR] No SNMOT-* sequences found in train/")

    selected = ([TRAIN_DIR / s for s in args.seq] if args.seq else all_seqs)
    for s in selected:
        if not s.exists():
            sys.exit(f"[ERROR] Sequence not found: {s}")

    print("Territorial Polygon Formation + xG Analysis")
    print(f"Sequences : {len(selected)}   Output : {OUTPUT_DIR}")

    # Train global xG model using dimensions from the first available sequence
    seq_W, seq_H = read_seq_dims(all_seqs[0])
    build_xg_model(W=seq_W, H=seq_H)

    for seq_dir in selected:
        process_sequence(seq_dir, OUTPUT_DIR)

    print(f"\n✅  Done. Outputs → {OUTPUT_DIR}/")
    print("     *_territory.mp4          — annotated video")
    print("     *_territory.csv          — per-frame area + xG stats")
    print("     *_territory_xg_chart.png — scatter + time-series chart")


if __name__ == "__main__":
    main()

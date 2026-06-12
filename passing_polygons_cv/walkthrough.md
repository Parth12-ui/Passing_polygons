# Passing Polygon CV — Guide & Walkthrough

## What It Does

For each football sequence in `train/`, the pipeline:

1. **Reads** the pre-annotated player positions (bounding boxes) and ball position from [gt/gt.txt](file:///Users/parth/course_codes/dsac/passing_polygons_cv/train/SNMOT-060/gt/gt.txt)
2. **Identifies** each person's team (HOME / AWAY) or role (referee → ignored) from [gameinfo.ini](file:///Users/parth/course_codes/dsac/passing_polygons_cv/train/SNMOT-060/gameinfo.ini)
3. **Finds the ball carrier** — the field player closest to the ball in each frame
4. **Draws passing lines** from the ball carrier to every teammate, coloured by whether an opponent is in the way
5. **Overlays** everything onto the real match images from `img1/`
6. **Reassembles** the annotated frames into an MP4 at the original frame rate

---

## Sample Output

![Frame 200 — ball carrier with passing polygon](carrier_sample_200.jpg)

> **Ball carrier** is the HOME player on the left (yellow ring + "BALL" label). Green lines = open to those two teammates. Red lines from the AWAY carrier = all passes blocked by HOME players spread across the pitch.

---

## Visual Legend

| Element | Meaning |
|---|---|
| 🟦 **Amber box** | HOME team player (team left) |
| 🟥 **Red box** | AWAY team player (team right) |
| 🟡 **Yellow ring + "BALL"** | Ball carrier — the player currently with the ball |
| 🟡 **Yellow dot** | Ball position |
| 🟢 **Green line** | Open passing lane — no opponent in the path |
| 🔴 **Red line** | Blocked passing lane — at least one opponent between passer and receiver |
| 🔵 **Coloured dot** | Foot anchor / player node |

Lines radiate **only from the ball carrier** to each of their teammates. All other players are still drawn (boxes + node dots) so you can see the full field layout.

---

## How the Data Works

### Directory structure
```
train/
└── SNMOT-060/
    ├── img1/          ← Real match frames (000001.jpg … 000750.jpg)
    ├── gt/gt.txt      ← Bounding-box annotations per frame
    ├── gameinfo.ini   ← Maps each track ID to a role + jersey
    └── seqinfo.ini    ← FPS, resolution, sequence length
```

### [gt.txt](file:///Users/parth/course_codes/dsac/passing_polygons_cv/train/SNMOT-060/gt/gt.txt) format (MOT standard)
```
frame_id, track_id, x, y, width, height, confidence, ...
```
[(x, y)](file:///Users/parth/course_codes/dsac/passing_polygons_cv/main.py#427-452) is the **top-left** corner of the bounding box in pixels.

### [gameinfo.ini](file:///Users/parth/course_codes/dsac/passing_polygons_cv/train/SNMOT-060/gameinfo.ini) roles
| Role string | Treatment |
|---|---|
| `player team left` | HOME player ✅ |
| `player team right` | AWAY player ✅ |
| `goalkeeper team left/right` | Counted as a player ✅ |
| `referee` | **Ignored** |
| [ball](file:///Users/parth/course_codes/dsac/passing_polygons_cv/main.py#185-201) | Ball tracklet — used only for position |
| `crowd`, `ball boy` | **Ignored** |

---

## How Blocking Is Detected

For each candidate pass `carrier → teammate`, the code casts a straight-line segment between their **foot points** (bottom-centre of each bounding box).

Every opponent's bounding box is expanded by **12 px** to account for body width, then tested for intersection with that segment. If any opponent box intersects → the lane is **blocked** (red). Otherwise → **open** (green).

---

## Ball Carrier Detection

Each frame, the pipeline computes the **Euclidean distance** from the ball's centre to every field player's foot point. The closest player — regardless of team — is designated the ball carrier for that frame.

---

## How to Run

### Requirements
```bash
pip install opencv-python numpy
```

### Process all 57 sequences
```bash
cd /Users/parth/course_codes/dsac/passing_polygons_cv
python main.py
```
Output MP4s saved to `output/<SNMOT-XXX>_passing.mp4`.

### Process specific sequences
```bash
python main.py --seq SNMOT-060 SNMOT-070
```

### Output location
```
passing_polygons_cv/
└── output/
    ├── SNMOT-060_passing.mp4
    ├── SNMOT-061_passing.mp4
    └── ...
```

Each MP4 is encoded with the `mp4v` codec at **25 fps** and the original resolution (**1920 × 1080**).

---

## Key Parameters (in [main.py](file:///Users/parth/course_codes/dsac/passing_polygons_cv/main.py))

| Constant | Default | Effect |
|---|---|---|
| `BLOCK_EXPAND_PX` | `12` | How many px to expand opponent bbox for blocking test |
| `MIN_PASS_DIST` | `30` | Minimum distance (px) before a passing line is drawn |
| `LINE_ALPHA` | `0.75` | Opacity of passing lines blended onto the frame |
| `NODE_RADIUS` | `5` | Size of the foot-anchor circle |
| `CARRIER_RING_RADIUS` | `14` | Size of the ball-carrier highlight ring |

Tune these at the top of [main.py](file:///Users/parth/course_codes/dsac/passing_polygons_cv/main.py) to adjust the visual density and sensitivity.

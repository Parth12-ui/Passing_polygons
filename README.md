# Passing Triangles Visualizer — SkillCorner

An interactive, browser-based football analytics tool that visualizes **passing options**, **pressure detection**, and **player movement** for every possession event across 10 real SkillCorner matches.

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [How to Run](#how-to-run)
3. [What We Did — Full Explanation](#what-we-did--full-explanation)
   - [Step 1 — Data Preprocessing (`preprocess.py`)](#step-1--data-preprocessing-preprocesspy)
   - [Step 2 — Blocking Detection Algorithm](#step-2--blocking-detection-algorithm)
   - [Step 3 — Pass Quality Classification](#step-3--pass-quality-classification)
   - [Step 4 — Player Snapping (Timeline)](#step-4--player-snapping-timeline)
   - [Step 5 — Browser Visualization (`index.html`)](#step-5--browser-visualization-indexhtml)
   - [Step 6 — Coordinate System](#step-6--coordinate-system)
   - [Step 7 — Passing Triangle Chains](#step-7--passing-triangle-chains)
   - [Step 8 — Interactive Features](#step-8--interactive-features)
4. [UI Controls Reference](#ui-controls-reference)
5. [Color Legend](#color-legend)

---

## Project Structure

```
Passing_polygons/
├── preprocess.py       ← Python script: reads raw data, writes data.json
├── index.html          ← Single-page browser app (Canvas + vanilla JS)
├── data.json           ← Generated output (~35 MB), consumed by index.html
└── opendata-master/
    └── data/
        ├── matches.json          ← Index of all 10 matches
        └── matches/
            └── <match_id>/
                ├── <id>_match.json           ← Rosters, team kits, pitch size
                └── <id>_dynamic_events.csv   ← Frame-level tracking events
```

---

## How to Run

### Prerequisites

- Python 3.8+
- A modern web browser (Chrome / Firefox / Edge)
- No external Python packages are required — only the standard library (`json`, `csv`, `os`, `math`, `collections`)

### Step 1 — (Optional) Re-generate `data.json`

> **Skip this step** if `data.json` already exists in the project root. It's already pre-built and ready to use.

```bash
cd /Users/parth/course_codes/dsac/Passing_polygons
python3 preprocess.py
```

This reads all match CSVs from `opendata-master/data/` and writes a single `data.json` (~35 MB) to the project root. Expect it to take **20–60 seconds** depending on your machine.

### Step 2 — Serve the files locally

`index.html` uses `fetch('data.json')`, which requires an HTTP server (browsers block local `file://` fetches for security). Run:

```bash
cd /Users/parth/course_codes/dsac/Passing_polygons
python3 -m http.server 8080
```

### Step 3 — Open in browser

```
http://localhost:8080
```

You'll see the **Passing Triangles** dashboard load immediately.

> **Tip:** Use keyboard arrow keys (`←` / `→`) to step through events one by one.

---

## What We Did — Full Explanation

### Step 1 — Data Preprocessing (`preprocess.py`)

The raw SkillCorner dataset provides two files per match:

| File | Content |
|---|---|
| `<id>_match.json` | Team info, kit colors, player roster with positions |
| `<id>_dynamic_events.csv` | Row-per-frame tracking data: player positions, speeds, angles, passing options |

The preprocessor:

1. **Loads `matches.json`** — an index listing all 10 match IDs, home/away teams, and dates.
2. **For each match**, calls `process_match()` which:
   - Loads the match JSON for rosters and team metadata.
   - Loads the CSV which can have **tens of thousands of rows**.
   - Separates rows by `event_type` into two buckets:
     - `player_possession` — a player has the ball right now
     - `passing_option` — a teammate that *could* receive a pass at this moment, linked to a possession event via `associated_player_possession_event_id`
3. **Builds a frame-level position dictionary** `frame_player_dict[frame][player_id]` by scanning every row for `x_start`, `y_start`, `trajectory_angle`, and `speed_avg`. This captures the exact location and movement of every tracked player at every frame.
4. **Builds a `player_timeline`** — a sorted list of `(frame, x, y, angle, speed, team_id)` entries per player, used for position snapping.
5. **Outputs `data.json`** — a compact JSON file containing one structured object per match, with all possession events, their passing options, opponent positions, and player rosters.

---

### Step 2 — Blocking Detection Algorithm

For each passing option, we check whether any **opposing player** is physically blocking the pass lane using a **vector projection** approach:

```
passer ──────────────────────► option player
              ↑
        opponent here?
```

**How it works:**

1. Compute the **pass vector** from passer `(pip_x, pip_y)` to option player `(opt_x, opt_y)`.
2. For every opposing player in the same frame:
   - Project the opponent's position onto the pass line using the dot product formula to find `t` (how far along the line the closest point is).
   - Clamp `t` to `[0.05, 0.95]` so we only consider opponents *between* passer and receiver, not behind either.
   - Compute `perp_dist` — the perpendicular distance from the opponent to the pass line.
3. If `perp_dist ≤ 2.5 meters`, the opponent is **in the lane**.
4. Additionally check if the opponent is **facing the pass direction** — compute the angular difference between the opponent's `trajectory_angle` and the pass direction. If within **±60°**, the opponent is actively oriented toward the ball → `is_facing = True`.
5. An opponent that `is_facing` while in the lane sets `movement_contested = True`.

This is more realistic than a simple circular zone: an opponent running *away* from the ball is not a meaningful blocker.

---

### Step 3 — Pass Quality Classification

Each passing option is labelled with one of three quality levels:

| Label | Color | Criteria |
|---|---|---|
| `blocked` | 🔴 Red | ≥1 facing opponent in lane, OR xPass < 40% AND (dangerous or difficult) |
| `clear` | 🟢 Green | xPass ≥ 65% AND no movement-contested opponent |
| `contested` | 🟠 Orange | Everything in between |

**xPass completion** (`xpass_completion`) is SkillCorner's pre-computed metric (0.0–1.0) estimating the probability a pass to that player succeeds, based on positions, distances, and historical data. We use it as the primary signal, augmented by our movement-blocking detection.

---

### Step 4 — Player Snapping (Timeline)

A single possession event may only have data for a handful of players who touched the ball at that exact frame. But we want to show **all roster players** on the pitch.

The `snap_position()` function binary-searches the sorted `player_timeline` for each roster player and finds their **last known position at or before the current frame**. This gives a realistic spatial snapshot without requiring every player to have an event at every frame.

```python
def snap_position(roster_pid, before_frame):
    # Binary search player_timeline[roster_pid] for last frame <= before_frame
    ...
    return best  # (frame, x, y, angle, speed, team_id)
```

The snapped positions are saved in `opponent_positions` and `teammate_positions` lists on each event object in `data.json`.

---

### Step 5 — Browser Visualization (`index.html`)

The entire frontend is a **single HTML file** with no frameworks or build tools. It uses:

- **HTML5 Canvas** for all pitch and player rendering
- **Vanilla JavaScript** for state management, event navigation, and interaction
- **CSS Grid + Flexbox** for the layout (pitch | right panel, with a timeline strip at bottom)
- **Google Fonts** (Inter, Roboto Mono) for typography

On page load, `boot()` fetches `data.json`, populates the match dropdown, and renders the first event of the first match.

The main `render()` function is called on every navigation step and draws in layer order:

```
Layer 1: Pitch background (gradient + stripes + field markings)
Layer 2: All opponent players (circles + movement arrows)
Layer 3: All non-ball-holder teammates (semi-transparent circles)
Layer 4: Passing option arrows (passer → each option, color-coded)
Layer 5: Blocking opponent rings (orange/red halos on blockers)
Layer 6: Triangle chain dashed lines (secondary passing connections)
Layer 7: Option player circles (drawn on top of arrows)
Layer 8: Intercepted pass or actual pass line
Layer 9: Ball holder (brightest, with glowing ring + ball marker)
```

---

### Step 6 — Coordinate System

SkillCorner uses a **pitch-center origin** where:
- `x` runs along the pitch length, from `-PL/2` to `+PL/2` (e.g. -52.5 to 52.5)
- `y` runs along the pitch width, from `-PW/2` to `+PW/2` (e.g. -34 to 34)
- Positive `y` = left side of pitch

The canvas uses a **top-left origin**. The conversion function `p2c(sx, sy)` maps SkillCorner coordinates to canvas pixels:

```js
function p2c(sx, sy) {
  return [
    (sx / PL + 0.5) * canvas.width,   // shift right by half
    (-sy / PW + 0.5) * canvas.height  // flip Y axis
  ];
}
```

Player movement arrows (`moveArrow`) use `trajectory_angle` in degrees and scale the arrow length by speed, capped at 28px to avoid clutter.

---

### Step 7 — Passing Triangle Chains

Beyond showing individual pass options, we draw **secondary connections** between option players who are close enough to pass to each other (≤ 28 meters apart). These are the "chains" in the ⛓ Chains toggle.

```js
function findTrianglePairs(opts) {
  // Returns all pairs [i, j] where distance(opts[i], opts[j]) <= 28m
}
```

These are drawn as **dashed white lines** with a small midpoint arrowhead, visualizing combinatorial passing networks — i.e., if I pass to player A, can A immediately pass to player B? This forms the "polygon/triangle" shape the feature is named after.

---

### Step 8 — Interactive Features

| Feature | How It Works |
|---|---|
| **Event navigation** | `◂` / `▸` buttons (or `←` / `→` keys) change `S.ei` (event index) and re-render |
| **Timeline slider** | Dragging jumps directly to any event by index |
| **Match selector** | Switching match reloads `S.match` and resets filters |
| **Period / Team filters** | `applyFilters()` re-filters `S.events` from the match's full event list |
| **Option hover** | Hovering a row in the right panel sets `S.hovOpt`, re-renders the canvas (highlighted arrow), shows tooltip |
| **Player hover (canvas)** | Mouse move does hit-testing against `S.drawnPlayers` (radius 18px), shows per-player cumulative stats tooltip |
| **Cumulative player stats** | `computePlayerStats()` replays all events up to the current index and accumulates possessions, passes, xPass, option quality counts per player |
| **Toggle buttons** | ⛓ Chains, 🔴 Opponents, 🛡 Blockers, ✂ Intercepts — each toggles a `S.show*` flag and re-renders |

---

## UI Controls Reference

| Control | Description |
|---|---|
| Match dropdown | Select one of the 10 matches |
| `All / P1 / P2` | Filter by half (All = whole game) |
| `Both Teams / Home / Away` | Filter whose possession events are shown |
| `⛓ Chains` | Toggle dashed secondary passing connection lines |
| `🔴 Opp.` | Toggle opponent player rendering |
| `🛡 Block` | Toggle blocking opponent highlight rings |
| `✂ Intercept` | Toggle intercepted pass visualization |
| Timeline `◂ ▸` | Step through events one at a time |
| Keyboard `←` `→` | Same as above |
| Slider | Jump to any event |
| **Hover player (canvas)** | Shows cumulative match stats tooltip |
| **Hover option (sidebar)** | Highlights arrow on pitch, shows xPass / xThreat / distance |

---

## Color Legend

| Color | Meaning |
|---|---|
| 🟢 Green arrow | Clear pass — high xPass%, no contested blocker |
| 🟠 Orange arrow | Contested pass — moderate risk |
| 🔴 Red arrow | Blocked / high-risk pass |
| 🟣 Purple dashed line | Intercepted pass trajectory |
| White solid line | Actual completed pass |
| Dashed white line | Triangle chain (secondary passing connection) |
| Red/orange ring on opponent | Blocking opponent in pass lane (red = facing, orange = in lane) |
| `🎯` icon | This was the actual pass target |
| `↑` icon | This pass would break a defensive line |
| `⚠` icon | Dangerous pass option |
| White dot above player | Player currently has the ball |
| Movement arrow | Player's current speed and direction of travel |

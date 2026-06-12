"""
preprocess.py
Converts SkillCorner open data (dynamic_events CSV + match JSON) for all 10 matches
into a single data.json suitable for the browser-based passing triangle visualization.
"""

import json
import csv
import os
import math
from collections import defaultdict

DATA_DIR = os.path.join(os.path.dirname(__file__), "opendata-master", "data")
MATCHES_JSON = os.path.join(DATA_DIR, "matches.json")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "data.json")

# ---- helpers ----

def safe_float(v, default=None):
    try:
        return float(v) if v not in (None, "", "nan", "NaN") else default
    except (TypeError, ValueError):
        return default

def safe_int(v, default=None):
    try:
        return int(float(v)) if v not in (None, "", "nan", "NaN") else default
    except (TypeError, ValueError):
        return default

def safe_bool(v):
    if isinstance(v, bool): return v
    if isinstance(v, str): return v.strip().lower() in ("true", "1", "yes")
    return bool(v) if v is not None else False


def load_match_info(match_id):
    path = os.path.join(DATA_DIR, "matches", str(match_id), f"{match_id}_match.json")
    with open(path, "r") as f:
        return json.load(f)


def load_events(match_id):
    path = os.path.join(DATA_DIR, "matches", str(match_id), f"{match_id}_dynamic_events.csv")
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def load_tracking(match_id):
    """
    Reads {match_id}_tracking_extrapolated.jsonl and builds:
      {frame -> {player_id(str) -> (x, y)}}
    Returns empty dict if file is missing or is still an LFS pointer.
    """
    path = os.path.join(DATA_DIR, "matches", str(match_id), f"{match_id}_tracking_extrapolated.jsonl")
    if not os.path.exists(path):
        return {}
    tracking = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("version https://git-lfs"):
                return {}  # LFS pointer — not downloaded
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            frame = obj.get("frame")
            if frame is None:
                continue
            players = obj.get("player_data") or []
            if not players:
                continue
            frame_map = {}
            for p in players:
                pid = p.get("player_id")
                x = p.get("x")
                y = p.get("y")
                if pid is not None and x is not None and y is not None:
                    frame_map[str(pid)] = (x, y)
            tracking[frame] = frame_map
    return tracking


def build_player_map(match_info):
    """Returns {player_id -> {id, name, short_name, team_id, position, number}}"""
    pmap = {}
    for p in match_info.get("players", []):
        pid = str(p["id"])
        pmap[pid] = {
            "id": p["id"],
            "name": p.get("short_name") or f"{p.get('first_name','')} {p.get('last_name','')}".strip(),
            "team_id": p["team_id"],
            "position": p.get("player_role", {}).get("acronym", "?"),
            "position_name": p.get("player_role", {}).get("name", ""),
            "number": p.get("number"),
            "trackable_object": p.get("trackable_object"),
        }
    return pmap


def build_team_info(match_info):
    """Returns {team_id -> {id, name, short_name, color, number_color, side}}"""
    home_team = match_info["home_team"]
    away_team = match_info["away_team"]
    home_kit = match_info.get("home_team_kit", {})
    away_kit = match_info.get("away_team_kit", {})
    home_sides = match_info.get("home_team_side", [])  # e.g. ["right_to_left", "left_to_right"]

    teams = {
        home_team["id"]: {
            "id": home_team["id"],
            "name": home_team.get("name", home_team.get("short_name", "")),
            "short_name": home_team.get("short_name", ""),
            "acronym": home_team.get("acronym", ""),
            "color": home_kit.get("jersey_color", "#1a56db"),
            "number_color": home_kit.get("number_color", "#ffffff"),
            "side": "home",
            "attack_direction_p1": home_sides[0] if home_sides else "left_to_right",
            "attack_direction_p2": home_sides[1] if len(home_sides) > 1 else "right_to_left",
        },
        away_team["id"]: {
            "id": away_team["id"],
            "name": away_team.get("name", away_team.get("short_name", "")),
            "short_name": away_team.get("short_name", ""),
            "acronym": away_team.get("acronym", ""),
            "color": away_kit.get("jersey_color", "#e02020"),
            "number_color": away_kit.get("number_color", "#ffffff"),
            "side": "away",
            "attack_direction_p1": home_sides[1] if len(home_sides) > 1 else "right_to_left",
            "attack_direction_p2": home_sides[0] if home_sides else "left_to_right",
        },
    }
    return teams


def angle_diff(a1, a2):
    """Smallest angular difference between two angles in degrees."""
    diff = abs(a1 - a2) % 360
    return diff if diff <= 180 else 360 - diff


def process_match(match_id, match_info_from_list):
    """Returns structured match data dict ready for JSON output."""
    print(f"  Processing match {match_id}...")

    try:
        match_info = load_match_info(match_id)
    except Exception as e:
        print(f"    ERROR loading match info: {e}")
        return None

    player_map = build_player_map(match_info)
    teams = build_team_info(match_info)
    pitch_length = match_info.get("pitch_length", 105)
    pitch_width = match_info.get("pitch_width", 68)

    home_team_id = match_info["home_team"]["id"]
    away_team_id = match_info["away_team"]["id"]

    try:
        all_events = load_events(match_id)
    except Exception as e:
        print(f"    ERROR loading events: {e}")
        return None

    print(f"    Loaded {len(all_events)} events")

    # Load frame-level tracking data (exact positions for all players at every frame)
    print(f"    Loading tracking data...")
    tracking = load_tracking(match_id)
    has_tracking = bool(tracking)
    print(f"    Tracking frames loaded: {len(tracking)}" if has_tracking else "    No tracking data — using event-based snap fallback")

    # -------------------------------------------------------------------------
    # Group events: find player_possession events + their linked passing_option
    # passing_option row: associated_player_possession_event_id links to the
    # player_possession event_id they belong to
    # -------------------------------------------------------------------------
    possessions = {}  # event_id -> row dict
    options_by_possession = defaultdict(list)  # event_id -> [option_row, ...]
    all_player_possession_frames = {}  # frame_start -> list of player rows (for opponent positions)
    frame_player_positions = defaultdict(list)  # frame_start -> list of {player, x, y, team, angle, speed}

    for row in all_events:
        etype = row.get("event_type", "")
        if etype == "player_possession":
            eid = row.get("event_id", "")
            possessions[eid] = row
        elif etype == "passing_option":
            linked = row.get("associated_player_possession_event_id", "")
            if linked:
                options_by_possession[linked].append(row)

    # Build frame-level player positions from all events for opponent detection.
    # We collect positions from ALL event rows (player_possession, passing_option, etc.)
    # and also from the player_in_possession_x/y fields in passing_option rows.
    # Use a dict {frame -> {player_id -> data}} to deduplicate.
    frame_player_dict = defaultdict(dict)  # {frame -> {player_id -> {...}}}

    for row in all_events:
        frame = safe_int(row.get("frame_start"))
        if frame is None:
            continue

        # Primary player position (the subject of this event row)
        x = safe_float(row.get("x_start"))
        y = safe_float(row.get("y_start"))
        pid = row.get("player_id", "")
        tid = row.get("team_id", "")
        angle = safe_float(row.get("trajectory_angle"))
        speed = safe_float(row.get("speed_avg"))
        if pid and tid and x is not None and y is not None:
            frame_player_dict[frame][pid] = {
                "player_id": pid,
                "team_id": tid,
                "x": x,
                "y": y,
                "angle": angle,
                "speed": speed,
            }

        # For passing_option rows: also capture player_in_possession position
        if row.get("event_type") == "passing_option":
            pip_id = row.get("player_in_possession_id", "")
            pip_tid = row.get("team_id", "")  # pip is always same team as option's team? No - use player_in_possession info
            # The pip_tid: passing_option row's team_id is the OPTION player's team
            # player_in_possession belongs to the SAME team as the option player (teammates)
            pip_x = safe_float(row.get("player_in_possession_x_start"))
            pip_y = safe_float(row.get("player_in_possession_y_start"))
            if pip_id and pip_x is not None and pip_y is not None:
                # team_id for pip: same as option player since they're teammates
                frame_player_dict[frame][pip_id] = {
                    "player_id": pip_id,
                    "team_id": tid,  # same team as option (teammates)
                    "x": pip_x,
                    "y": pip_y,
                    "angle": None,
                    "speed": None,
                }

    # Flatten back to list format
    for frame, players_by_id in frame_player_dict.items():
        frame_player_positions[frame] = list(players_by_id.values())

    # ---- Build per-player sorted timeline ----
    # {player_id -> [(frame, x, y, angle, speed, team_id), ...]} sorted by frame asc
    player_timeline = defaultdict(list)
    for frame, players in frame_player_positions.items():
        for fp in players:
            player_timeline[fp["player_id"]].append(
                (frame, fp["x"], fp["y"], fp["angle"], fp["speed"], fp["team_id"])
            )
    for pid in player_timeline:
        player_timeline[pid].sort(key=lambda t: t[0])

    # -------------------------------------------------------------------------
    # Build possession events list
    # -------------------------------------------------------------------------
    events_out = []

    for eid, prow in possessions.items():
        period = safe_int(prow.get("period", 1), 1)
        frame_start = safe_int(prow.get("frame_start"))
        minute_start = safe_int(prow.get("minute_start"))
        second_start = safe_int(prow.get("second_start"))
        time_start = prow.get("time_start", "")
        end_type = prow.get("end_type", "")
        pass_outcome = prow.get("pass_outcome", "")

        # possessing player
        pid = str(prow.get("player_id", ""))
        pname = prow.get("player_name", "Unknown")
        tid = str(prow.get("team_id", ""))
        px = safe_float(prow.get("x_start"))
        py = safe_float(prow.get("y_start"))
        px_end = safe_float(prow.get("x_end"))
        py_end = safe_float(prow.get("y_end"))
        p_angle = safe_float(prow.get("trajectory_angle"))
        p_speed = safe_float(prow.get("speed_avg"))
        p_direction = prow.get("trajectory_direction", "")

        # pass target (where they actually passed to, if they made a pass)
        targeted_id = str(prow.get("player_targeted_id", ""))
        targeted_name = prow.get("player_targeted_name", "")
        targeted_x_pass = safe_float(prow.get("player_targeted_x_pass"))
        targeted_y_pass = safe_float(prow.get("player_targeted_y_pass"))
        targeted_x_rec = safe_float(prow.get("player_targeted_x_reception"))
        targeted_y_rec = safe_float(prow.get("player_targeted_y_reception"))
        targeted_xpass = safe_float(prow.get("player_targeted_xpass_completion"))
        targeted_xthreat = safe_float(prow.get("player_targeted_xthreat"))
        n_passing_options = safe_int(prow.get("n_passing_options"))

        # determine if this is an intercepted pass
        is_intercepted = (
            pass_outcome.lower() in ("unsuccessful", "failure", "intercepted", "failed")
            or "intercept" in end_type.lower()
            or "intercept" in pass_outcome.lower()
        )

        # ---- passing options ----
        options_list = options_by_possession.get(eid, [])
        passing_options = []

        # Collect all players in this frame for opponent detection
        frame_players = frame_player_positions.get(frame_start, [])

        for opt in options_list:
            opt_pid = str(opt.get("player_id", ""))
            opt_name = opt.get("player_name", "Unknown")
            opt_tid = str(opt.get("team_id", ""))
            opt_x = safe_float(opt.get("x_start"))
            opt_y = safe_float(opt.get("y_start"))
            opt_angle = safe_float(opt.get("trajectory_angle"))
            opt_speed = safe_float(opt.get("speed_avg"))
            opt_direction = opt.get("trajectory_direction", "")

            # pass quality metrics
            dangerous = safe_bool(opt.get("dangerous", False))
            difficult = safe_bool(opt.get("difficult_pass_target", False))
            xthreat = safe_float(opt.get("xthreat"), 0.0)
            xpass_comp = safe_float(opt.get("xpass_completion"), 0.0)
            opt_score = safe_float(opt.get("passing_option_score"), 0.0)
            targeted_flag = safe_bool(opt.get("targeted", False))
            received_flag = safe_bool(opt.get("received", False))
            pass_dist = safe_float(opt.get("pass_distance"))
            pass_dir = opt.get("pass_direction", "")
            pass_angle_val = safe_float(opt.get("pass_angle"))
            n_opp_ahead = safe_int(opt.get("n_opponents_ahead_player_in_possession_pass_moment"))
            interplayer_dist = safe_float(opt.get("interplayer_distance"))
            line_break = safe_bool(opt.get("break_defensive_line", False))
            push_line = safe_bool(opt.get("push_defensive_line", False))
            received_in_space = safe_bool(opt.get("received_in_space", False))

            # position of ball holder at time of option
            pip_x = safe_float(opt.get("player_in_possession_x_start"))
            pip_y = safe_float(opt.get("player_in_possession_y_start"))

            # ---- Movement-aware blocking detection ----
            # Compute pass vector direction (from ball holder to option player)
            movement_contested = False
            blocking_opponents = []

            if pip_x is not None and pip_y is not None and opt_x is not None and opt_y is not None:
                pass_vec_x = opt_x - pip_x
                pass_vec_y = opt_y - pip_y
                pass_len = math.sqrt(pass_vec_x**2 + pass_vec_y**2)
                pass_direction_deg = math.degrees(math.atan2(pass_vec_y, pass_vec_x)) % 360

                # Check opposing players
                for fp in frame_players:
                    if fp["team_id"] == opt_tid:  # same team as option player = also opposing possession holder
                        continue
                    if fp["team_id"] == tid:  # same team as possession holder → not an opponent
                        continue
                    if fp["angle"] is None:
                        continue

                    # Project opponent onto pass line
                    opp_vec_x = fp["x"] - pip_x
                    opp_vec_y = fp["y"] - pip_y

                    if pass_len > 0:
                        t = (opp_vec_x * pass_vec_x + opp_vec_y * pass_vec_y) / (pass_len ** 2)
                        t = max(0.0, min(1.0, t))
                        proj_x = pip_x + t * pass_vec_x
                        proj_y = pip_y + t * pass_vec_y
                        perp_dist = math.sqrt((fp["x"] - proj_x)**2 + (fp["y"] - proj_y)**2)

                        # Opponent is within 2.5m of pass lane and between passer and target
                        if perp_dist <= 2.5 and 0.05 < t < 0.95:
                            # Check if facing the pass direction
                            opp_angle = fp["angle"] % 360
                            facing_diff = angle_diff(opp_angle, pass_direction_deg)
                            # Within ±60° = facing roughly toward pass
                            is_facing = facing_diff <= 60

                            blocking_opponents.append({
                                "player_id": fp["player_id"],
                                "x": fp["x"],
                                "y": fp["y"],
                                "angle": fp["angle"],
                                "t_along_pass": round(t, 3),
                                "perp_dist": round(perp_dist, 2),
                                "is_facing": is_facing,
                            })
                            if is_facing:
                                movement_contested = True

            # Classify pass quality
            # xpass_completion is stored as a decimal (0.0–1.0), avg ~0.77 in this dataset.
            # n_opponents_ahead is typically 4–10 (counts ALL opponents between passer+target).
            # movement_contested = True only when an opponent is actively facing the pass lane.
            #
            # Rules (calibrated from data distribution):
            #   "clear": xpass >= 0.65 AND not movement_contested
            #   "blocked": movement_contested (facing opponent in lane) OR
            #              (xpass < 0.40 AND (dangerous OR difficult))
            #   "contested": everything in between

            xp = xpass_comp if xpass_comp is not None else 0.5

            facing_count = sum(1 for b in blocking_opponents if b.get("is_facing"))

            if facing_count >= 1 or (xp < 0.40 and (dangerous or difficult)):
                quality = "blocked"
            elif xp >= 0.65 and not movement_contested:
                quality = "clear"
            else:
                quality = "contested"

            passing_options.append({
                "player_id": opt_pid,
                "player_name": opt_name,
                "team_id": opt_tid,
                "x": opt_x,
                "y": opt_y,
                "move_angle": opt_angle,
                "move_speed": opt_speed,
                "move_direction": opt_direction,
                "dangerous": dangerous,
                "difficult": difficult,
                "xthreat": xthreat,
                "xpass_completion": xpass_comp,
                "option_score": opt_score,
                "targeted": targeted_flag,
                "received": received_flag,
                "pass_distance": pass_dist,
                "pass_direction": pass_dir,
                "pass_angle": pass_angle_val,
                "n_opponents_ahead": n_opp_ahead,
                "interplayer_distance": interplayer_dist,
                "line_break": line_break,
                "push_defensive_line": push_line,
                "received_in_space": received_in_space,
                "quality": quality,
                "movement_contested": movement_contested,
                "blocking_opponents": blocking_opponents,
                "pip_x": pip_x,
                "pip_y": pip_y,
            })

        # ---- Snap ALL roster players to their last known position <= frame_start ----
        # This ensures every player on the pitch is visible, not just those with events at this frame.
        def snap_position(roster_pid, before_frame):
            """Return last known position of player at or before before_frame."""
            history = player_timeline.get(str(roster_pid), [])
            if not history:
                return None
            # Binary-search for last entry with frame <= before_frame
            lo, hi = 0, len(history) - 1
            best = None
            while lo <= hi:
                mid = (lo + hi) // 2
                if history[mid][0] <= (before_frame or 0):
                    best = history[mid]
                    lo = mid + 1
                else:
                    hi = mid - 1
            return best  # (frame, x, y, angle, speed, team_id)

        # ---- Build opponent_positions and teammate_positions ----
        # Prefer exact tracking frame; fall back to snap_position from event-based timeline.
        opponent_positions = []
        teammate_positions = []
        opp_team_id = str(away_team_id) if tid == str(home_team_id) else str(home_team_id)

        if has_tracking and frame_start is not None and frame_start in tracking:
            # Use exact tracking positions — all 22 players at this exact frame
            track_frame = tracking[frame_start]
            for roster_player in player_map.values():
                rp_id = str(roster_player["id"])
                rp_tid = str(roster_player["team_id"])
                if rp_id == pid:
                    continue  # ball holder drawn separately
                pos = track_frame.get(rp_id)
                if pos is None:
                    continue
                entry = {
                    "player_id": rp_id,
                    "x": pos[0],
                    "y": pos[1],
                    "angle": None,   # tracking data has no angle
                    "speed": None,
                }
                if rp_tid == opp_team_id:
                    opponent_positions.append(entry)
                else:
                    teammate_positions.append(entry)
        else:
            # Fallback: snap each roster player to last known event position
            for roster_player in player_map.values():
                rp_id = str(roster_player["id"])
                rp_tid = str(roster_player["team_id"])
                if rp_id == pid:
                    continue
                pos = snap_position(rp_id, frame_start)
                if pos is None:
                    continue
                entry = {
                    "player_id": rp_id,
                    "x": pos[1],
                    "y": pos[2],
                    "angle": pos[3],
                    "speed": pos[4],
                }
                if rp_tid == opp_team_id:
                    opponent_positions.append(entry)
                else:
                    teammate_positions.append(entry)

        event_obj = {
            "event_id": eid,
            "frame_start": frame_start,
            "period": period,
            "minute_start": minute_start,
            "second_start": second_start,
            "time_start": time_start,
            "end_type": end_type,
            "pass_outcome": pass_outcome,
            "is_pass": end_type in ("pass", "pass_reception"),
            "is_intercepted": is_intercepted,
            "player": {
                "id": pid,
                "name": pname,
                "team_id": tid,
                "x": px,
                "y": py,
                "x_end": px_end,
                "y_end": py_end,
                "move_angle": p_angle,
                "move_speed": p_speed,
                "move_direction": p_direction,
            },
            "targeted_player": {
                "id": targeted_id,
                "name": targeted_name,
                "x_pass": targeted_x_pass,
                "y_pass": targeted_y_pass,
                "x_reception": targeted_x_rec,
                "y_reception": targeted_y_rec,
                "xpass_completion": targeted_xpass,
                "xthreat": targeted_xthreat,
            } if targeted_id else None,
            "n_passing_options": n_passing_options,
            "passing_options": passing_options,
            "opponent_positions": opponent_positions,
            "teammate_positions": teammate_positions,
        }
        events_out.append(event_obj)

    # Sort events by period, then minute, then second
    events_out.sort(key=lambda e: (e["period"] or 1, e["minute_start"] or 0, e["second_start"] or 0))

    player_list = [dict(v, id=str(v["id"]), team_id=str(v["team_id"])) for v in player_map.values()]

    return {
        "match_id": match_id,
        "home_team_id": str(home_team_id),
        "away_team_id": str(away_team_id),
        "home_team_score": match_info.get("home_team_score"),
        "away_team_score": match_info.get("away_team_score"),
        "competition": match_info.get("competition_edition", {}).get("name", ""),
        "date": match_info_from_list.get("date_time", ""),
        "pitch_length": pitch_length,
        "pitch_width": pitch_width,
        "teams": {str(k): v for k, v in teams.items()},
        "players": player_list,
        "events": events_out,
    }


def main():
    print("Loading matches index...")
    with open(MATCHES_JSON, "r") as f:
        matches_index = json.load(f)

    all_matches = []
    for m in matches_index:
        match_id = m["id"]
        match_data = process_match(match_id, m)
        if match_data:
            all_matches.append(match_data)
            n_events = len(match_data["events"])
            # Show some stats
            n_options = sum(len(e["passing_options"]) for e in match_data["events"])
            print(f"    → {n_events} possession events, {n_options} passing options")

    output = {
        "matches": all_matches,
        "matches_index": [
            {
                "id": m["id"],
                "date": m["date_time"],
                "home": m["home_team"]["short_name"],
                "away": m["away_team"]["short_name"],
            }
            for m in matches_index
        ],
    }

    print(f"\nWriting {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, separators=(",", ":"))

    size_mb = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
    print(f"Done! data.json is {size_mb:.1f} MB")


if __name__ == "__main__":
    main()

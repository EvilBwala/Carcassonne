"""game_visualizer.py – Procedural Matplotlib visualizer for Carcassonne.

Plays a full random 2-player game starting at (0,0), recording every
completed turn with a detailed action log (tile placement, meeple
decisions, auto-removals from completed features, and sanity checks).
Renders an animated MP4 (or GIF fallback) with the board and a text
panel showing every step.

Usage
-----
    python game_visualizer.py                         # default random game
    python game_visualizer.py --seed 42 --fps 3       # reproducible, 3 fps
    python game_visualizer.py --seed 42 --frames      # also save PNGs
"""

import argparse
import os
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle, Polygon, Rectangle
import matplotlib.animation as animation

from carcassonne.carcassonne_game_state import CarcassonneGameState
from carcassonne.objects.actions.meeple_action import MeepleAction
from carcassonne.objects.actions.pass_action import PassAction
from carcassonne.objects.actions.tile_action import TileAction
from carcassonne.objects.coordinate import Coordinate
from carcassonne.objects.coordinate_with_side import CoordinateWithSide
from carcassonne.objects.game_phase import GamePhase
from carcassonne.objects.meeple_position import MeeplePosition
from carcassonne.objects.meeple_type import MeepleType
from carcassonne.objects.side import Side
from carcassonne.objects.terrain_type import TerrainType
from carcassonne.objects.tile import Tile
from carcassonne.tile_sets.supplementary_rules import SupplementaryRule
from carcassonne.tile_sets.tile_sets import TileSet
from carcassonne.utils.action_util import ActionUtil
from carcassonne.utils.city_util import CityUtil
from carcassonne.utils.points_collector import PointsCollector
from carcassonne.utils.road_util import RoadUtil
from carcassonne.utils.state_updater import StateUpdater

# ═══════════════════════════════════════════════════════════════════════
# Colour palette
# ═══════════════════════════════════════════════════════════════════════
GRASS_CLR     = "#6a994e"
CITY_CLR      = "#bc6c25"
CITY_EDGE_CLR = "#7f4f24"
ROAD_CLR      = "#d4d4d4"
ROAD_EDGE_CLR = "#6b705c"
CHAPEL_BG     = "#ffffff"
CHAPEL_FG     = "#6b4226"
TILE_EDGE_CLR = "#333333"
HIGHLIGHT_CLR = "#ffd60a"
EMPTY_CLR     = "#e8e8e4"
BG_CLR        = "#f0ead2"
PLAYER_CLR    = ["#e63946", "#457b9d"]

# ═══════════════════════════════════════════════════════════════════════
# Name mappings
# ═══════════════════════════════════════════════════════════════════════
SIDE_NAME = {
    Side.TOP: "T", Side.TOP_RIGHT: "TR", Side.RIGHT: "R",
    Side.BOTTOM_RIGHT: "BR", Side.BOTTOM: "B", Side.BOTTOM_LEFT: "BL",
    Side.LEFT: "L", Side.TOP_LEFT: "TL", Side.CENTER: "C",
}
SIDE_FULL = {
    Side.TOP: "TOP", Side.RIGHT: "RIGHT", Side.BOTTOM: "BOTTOM",
    Side.LEFT: "LEFT", Side.CENTER: "CENTER", Side.TOP_LEFT: "TOP_LEFT",
    Side.TOP_RIGHT: "TOP_RIGHT", Side.BOTTOM_LEFT: "BOTTOM_LEFT",
    Side.BOTTOM_RIGHT: "BOTTOM_RIGHT",
}
MTYPE_NAME = {
    MeepleType.NORMAL: "normal", MeepleType.ABBOT: "abbot",
    MeepleType.FARMER: "farmer", MeepleType.BIG: "BIG",
    MeepleType.BIG_FARMER: "BIG_FARMER",
}
CARDINAL = [Side.TOP, Side.RIGHT, Side.BOTTOM, Side.LEFT]

# ═══════════════════════════════════════════════════════════════════════
# Geometry helpers  (local tile coordinate: (0,0)=BL  (1,1)=TR)
# ═══════════════════════════════════════════════════════════════════════
SIDE_POS = {
    Side.TOP: (0.50, 1.00), Side.TOP_RIGHT: (0.82, 0.82),
    Side.RIGHT: (1.00, 0.50), Side.BOTTOM_RIGHT: (0.82, 0.18),
    Side.BOTTOM: (0.50, 0.00), Side.BOTTOM_LEFT: (0.18, 0.18),
    Side.LEFT: (0.00, 0.50), Side.TOP_LEFT: (0.18, 0.82),
    Side.CENTER: (0.50, 0.50),
}

_SIDE_ORDER = [Side.TOP, Side.RIGHT, Side.BOTTOM, Side.LEFT]
_CORNERS    = [(0.0, 1.0), (1.0, 1.0), (1.0, 0.0), (0.0, 0.0)]

_DEPTH = 0.35
_INNER_START = {
    Side.TOP: (0.12, 1.0 - _DEPTH), Side.RIGHT: (1.0 - _DEPTH, 0.88),
    Side.BOTTOM: (0.88, _DEPTH), Side.LEFT: (_DEPTH, 0.12),
}
_INNER_END = {
    Side.TOP: (0.88, 1.0 - _DEPTH), Side.RIGHT: (1.0 - _DEPTH, 0.12),
    Side.BOTTOM: (0.12, _DEPTH), Side.LEFT: (_DEPTH, 0.88),
}
_ROAD_EDGE_POS = {
    Side.TOP: (0.50, 1.00), Side.RIGHT: (1.00, 0.50),
    Side.BOTTOM: (0.50, 0.00), Side.LEFT: (0.00, 0.50),
    Side.CENTER: (0.50, 0.50),
}


def _city_poly(city_group: list) -> list:
    side_set = set(city_group)
    if not side_set:
        return []
    is_city = [s in side_set for s in _SIDE_ORDER]
    pts: list = []
    for i in range(4):
        if not is_city[i]:
            continue
        side = _SIDE_ORDER[i]
        if not is_city[(i - 1) % 4]:
            pts.append(_INNER_START[side])
        pts.append(_CORNERS[i])
        pts.append(_CORNERS[(i + 1) % 4])
        if not is_city[(i + 1) % 4]:
            pts.append(_INNER_END[side])
    if not pts:
        return []
    clean = [pts[0]]
    for p in pts[1:]:
        if p != clean[-1]:
            clean.append(p)
    if len(clean) > 1 and clean[-1] == clean[0]:
        clean.pop()
    return clean


def _road_endpoints(conn):
    return (_ROAD_EDGE_POS.get(conn.a, (0.5, 0.5)),
            _ROAD_EDGE_POS.get(conn.b, (0.5, 0.5)))


def _tile_sides_str(t: Tile) -> str:
    parts = []
    for s in CARDINAL:
        tt = t.get_type(s)
        parts.append(f"{SIDE_NAME[s]}={tt.value if tt else '?'}")
    return " ".join(parts)


def _mp_desc(mp: MeeplePosition, player: int) -> str:
    cws = mp.coordinate_with_side
    return (f"P{player} {MTYPE_NAME.get(mp.meeple_type, '?')} "
            f"@ ({cws.coordinate.row},{cws.coordinate.column}) "
            f"{SIDE_FULL.get(cws.side, str(cws.side))}")


# ═══════════════════════════════════════════════════════════════════════
# Turn record
# ═══════════════════════════════════════════════════════════════════════
@dataclass
class TurnRecord:
    turn: int
    board: Dict[Tuple[int, int], Tile]
    scores: List[int]
    who_played: int
    action_log: List[str]
    warnings: List[str]
    placed_meeples: List[List[MeeplePosition]]
    last_placed: Optional[Tuple[int, int]]
    deck_remaining: int


# ═══════════════════════════════════════════════════════════════════════
# Random game loop with detailed logging + meeple checks
# ═══════════════════════════════════════════════════════════════════════
_VARIANTS = {
    "full": {
        "tile_sets": [TileSet.BASE, TileSet.INNS_AND_CATHEDRALS],
        "supplementary_rules": [SupplementaryRule.ABBOTS, SupplementaryRule.FARMERS],
    },
    "base": {
        "tile_sets": [TileSet.BASE],
        "supplementary_rules": [SupplementaryRule.FARMERS],
    },
}


def play_random_game(seed: Optional[int] = None,
                     max_turns: Optional[int] = None,
                     variant: str = "full") -> List[TurnRecord]:
    if seed is not None:
        random.seed(seed)

    if variant not in _VARIANTS:
        raise ValueError(f"Unknown variant '{variant}'. Choose from: {list(_VARIANTS.keys())}")
    cfg = _VARIANTS[variant]

    state = CarcassonneGameState(
        tile_sets=cfg["tile_sets"],
        supplementary_rules=cfg["supplementary_rules"],
        players=2,
        starting_position=Coordinate(0, 0),
    )

    records: List[TurnRecord] = []
    turn_num = 0
    cur_log: List[str] = []
    cur_warnings: List[str] = []
    last_placed: Optional[Tuple[int, int]] = None
    turn_player = 0

    while not state.is_terminated():
        actions = ActionUtil.get_possible_actions(state)
        if not actions:
            break

        action = random.choice(actions)
        old_phase = state.phase

        # ── Log the action BEFORE applying it ──────────────────────

        if isinstance(action, TileAction):
            turn_player = state.current_player
            coord = action.coordinate
            t = action.tile
            cur_log = [
                f"TILE: P{turn_player} places \"{t.description}\" "
                f"@ ({coord.row},{coord.column}) rot={action.tile_rotations}",
                f"  sides: [{_tile_sides_str(t)}]",
            ]
            cur_warnings = []
            last_placed = (coord.row, coord.column)

        elif isinstance(action, PassAction) and old_phase == GamePhase.TILES:
            turn_player = state.current_player
            cur_log = [f"TILE: P{turn_player} pass (tile unplayable, skipped)"]
            cur_warnings = []
            last_placed = None

        elif old_phase == GamePhase.MEEPLES:
            if isinstance(action, MeepleAction):
                cws = action.coordinate_with_side
                tile_at = state.board.get(
                    (cws.coordinate.row, cws.coordinate.column))
                terrain = tile_at.get_type(cws.side) if tile_at else None
                if terrain is None and action.meeple_type in (
                        MeepleType.FARMER, MeepleType.BIG_FARMER):
                    feature = "grass"
                else:
                    feature = terrain.value if terrain else "?"
                verb = "removes" if action.remove else "places"
                mname = MTYPE_NAME.get(action.meeple_type, "?")
                cur_log.append(
                    f"MEEPLE: P{state.current_player} {verb} {mname} "
                    f"on {feature} @ ({cws.coordinate.row},"
                    f"{cws.coordinate.column}) {SIDE_FULL.get(cws.side, '?')}"
                )
            elif isinstance(action, PassAction):
                cur_log.append(f"MEEPLE: P{state.current_player} pass (no meeple)")

        # ── Snapshot meeples + scores before meeple-phase resolution ──
        if old_phase == GamePhase.MEEPLES:
            pre_meeples = {p: set(state.placed_meeples[p])
                           for p in range(state.players)}
            pre_scores = list(state.scores)
            acting_player = state.current_player

            player_adds: Set[MeeplePosition] = set()
            player_removes: Set[MeeplePosition] = set()
            if isinstance(action, MeepleAction):
                mp = MeeplePosition(action.meeple_type,
                                    action.coordinate_with_side)
                if action.remove:
                    player_removes.add(mp)
                else:
                    player_adds.add(mp)

        # ── Apply the action ──────────────────────────────────────
        state = StateUpdater.apply_action(state, action)

        # ── After meeple-phase: detect auto-removals + verify ─────
        if old_phase == GamePhase.MEEPLES:
            post_meeples = {p: set(state.placed_meeples[p])
                            for p in range(state.players)}
            post_scores = list(state.scores)
            is_final = state.is_terminated()

            for p in range(state.players):
                expected = set(pre_meeples[p])
                if p == acting_player:
                    expected = (expected | player_adds) - player_removes
                auto_removed = expected - post_meeples[p]

                for mp in sorted(auto_removed,
                                 key=lambda m: (m.coordinate_with_side.coordinate.row,
                                                m.coordinate_with_side.coordinate.column)):
                    cws = mp.coordinate_with_side
                    row, col = cws.coordinate.row, cws.coordinate.column
                    side = cws.side
                    tile_at = state.board.get((row, col))
                    terrain = tile_at.get_type(side) if tile_at else None
                    if terrain is None and mp.meeple_type in (
                            MeepleType.FARMER, MeepleType.BIG_FARMER):
                        feature = "grass"
                    else:
                        feature = terrain.value if terrain else "?"
                    mname = MTYPE_NAME.get(mp.meeple_type, "?")

                    if is_final:
                        cur_log.append(
                            f"END-GAME: P{p} {mname} "
                            f"from {feature} @ ({row},{col}) "
                            f"{SIDE_FULL.get(side, '?')}"
                        )
                    else:
                        cur_log.append(
                            f"AUTO-REMOVE: P{p} {mname} "
                            f"from {feature} @ ({row},{col}) "
                            f"{SIDE_FULL.get(side, '?')}"
                        )
                        # Verify the feature is actually completed
                        _verify_removal(state, p, mp, terrain,
                                        row, col, side, cur_log, cur_warnings)

            # Score change
            gains = []
            for p in range(state.players):
                diff = post_scores[p] - pre_scores[p]
                if diff > 0:
                    gains.append(f"P{p}+{diff}")
            if gains:
                cur_log.append(f"SCORE: {', '.join(gains)}")

            # ── Record turn ───────────────────────────────────────
            turn_num += 1
            records.append(TurnRecord(
                turn=turn_num,
                board=dict(state.board),
                scores=list(state.scores),
                who_played=turn_player,
                action_log=list(cur_log),
                warnings=list(cur_warnings),
                placed_meeples=[list(pm) for pm in state.placed_meeples],
                last_placed=last_placed,
                deck_remaining=len(state.deck),
            ))
            if cur_warnings:
                for w in cur_warnings:
                    print(f"  ⚠ Turn {turn_num}: {w}")
            if max_turns is not None and turn_num >= max_turns:
                break

    return records


def _verify_removal(state, player, mp, terrain, row, col, side,
                    log, warnings):
    """Check that an auto-removed meeple was on a genuinely completed feature."""
    if terrain == TerrainType.CITY:
        city = CityUtil.find_city(
            state, CoordinateWithSide(Coordinate(row, col), side))
        n_tiles = len({p.coordinate for p in city.city_positions})
        if city.finished:
            log.append(f"  ✓ CHECK: city finished ({n_tiles} tiles)")
        else:
            warnings.append(
                f"BUG: P{player} meeple removed from UNFINISHED city "
                f"@ ({row},{col}) {SIDE_FULL.get(side, '?')}")
            log.append(f"  ✗ BUG: city NOT finished!")
    elif terrain == TerrainType.ROAD:
        road = RoadUtil.find_road(
            state, CoordinateWithSide(Coordinate(row, col), side))
        n_tiles = len({p.coordinate for p in road.road_positions})
        if road.finished:
            log.append(f"  ✓ CHECK: road finished ({n_tiles} tiles)")
        else:
            warnings.append(
                f"BUG: P{player} meeple removed from UNFINISHED road "
                f"@ ({row},{col}) {SIDE_FULL.get(side, '?')}")
            log.append(f"  ✗ BUG: road NOT finished!")
    elif terrain in (TerrainType.CHAPEL, TerrainType.FLOWERS):
        pts = PointsCollector.chapel_or_flowers_points(
            state, Coordinate(row, col))
        if pts == 9:
            log.append(f"  ✓ CHECK: chapel/flowers complete (9 tiles)")
        else:
            warnings.append(
                f"BUG: P{player} meeple removed from INCOMPLETE "
                f"chapel @ ({row},{col}), only {pts}/9 tiles")
            log.append(f"  ✗ BUG: chapel only {pts}/9!")
    elif terrain == TerrainType.GRASS:
        log.append(f"  ✓ CHECK: farmer — valid grass position")
    elif terrain is None:
        log.append(f"  ? CHECK: terrain=None (farmer on diagonal side)")


# ═══════════════════════════════════════════════════════════════════════
# Rendering
# ═══════════════════════════════════════════════════════════════════════
def _board_extent(records: List[TurnRecord]):
    rs, cs = [], []
    for rec in records:
        for r, c in rec.board:
            rs.append(r); cs.append(c)
    if not rs:
        return 0, 0, 0, 0
    return min(rs), max(rs), min(cs), max(cs)


def _draw_tile(ax, r, c, tile, *, highlight=False):
    x0, y0 = float(c), float(-(r + 1))
    ax.add_patch(Rectangle(
        (x0, y0), 1, 1, fc=GRASS_CLR, ec=TILE_EDGE_CLR, lw=1.0, zorder=1))
    for cg in tile.city:
        verts = _city_poly(cg)
        if len(verts) >= 3:
            ax.add_patch(Polygon(
                [(x0+px, y0+py) for px, py in verts], closed=True,
                fc=CITY_CLR, ec=CITY_EDGE_CLR, lw=0.7, zorder=2))
    for conn in tile.road:
        p1, p2 = _road_endpoints(conn)
        xs, ys = [x0+p1[0], x0+p2[0]], [y0+p1[1], y0+p2[1]]
        ax.plot(xs, ys, color=ROAD_CLR, lw=4.5, solid_capstyle="round", zorder=3)
        ax.plot(xs, ys, color=ROAD_EDGE_CLR, lw=1.0, solid_capstyle="round", zorder=3)
    if tile.chapel:
        cx, cy = x0+0.5, y0+0.5
        ax.add_patch(Circle((cx, cy), 0.10, fc=CHAPEL_BG, ec=CHAPEL_FG, lw=0.8, zorder=4))
        ax.plot(cx, cy, "+", color=CHAPEL_FG, ms=5, mew=1.2, zorder=4)
    if tile.shield:
        ax.text(x0+0.50, y0+0.50, "S", fontsize=8, ha="center", va="center",
                color="#ffb703", zorder=4, fontweight="bold", fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.05", fc=CITY_CLR, ec="#ffb703",
                          lw=0.6, alpha=0.85))
    if tile.flowers:
        ax.add_patch(Circle((x0+0.18, y0+0.18), 0.06,
                             fc="#fca311", ec="#e9c46a", lw=0.5, zorder=4))
    if highlight:
        ax.add_patch(Rectangle((x0-0.05, y0-0.05), 1.10, 1.10,
                                fc="none", ec=HIGHLIGHT_CLR, lw=2.8, zorder=5))
    ax.text(x0+0.06, y0+0.06, f"{r},{c}", fontsize=5, color="#333",
            alpha=0.5, zorder=6, fontfamily="monospace")


def _draw_meeple(ax, r, c, side, player, mtype):
    x0, y0 = float(c), float(-(r + 1))
    lx, ly = SIDE_POS.get(side, (0.5, 0.5))
    mx, my = x0+lx, y0+ly
    sz = 0.10 if mtype in (MeepleType.BIG, MeepleType.BIG_FARMER) else 0.07
    ax.add_patch(Circle((mx, my), sz, fc=PLAYER_CLR[player],
                         ec="white", lw=0.8, zorder=10))
    _LBL = {MeepleType.ABBOT: "A", MeepleType.FARMER: "F",
            MeepleType.BIG: "B", MeepleType.BIG_FARMER: "B"}
    lbl = _LBL.get(mtype)
    if lbl:
        ax.text(mx, my, lbl, fontsize=6, ha="center", va="center",
                color="white", zorder=11, fontweight="bold")


def render_frame(fig, ax, ax_log, rec: TurnRecord, extent: tuple):
    """Render a complete board state onto *ax* and action log onto *ax_log*."""
    ax.clear()
    ax_log.clear()
    min_r, max_r, min_c, max_c = extent
    pad = 1.5
    xl = min_c - pad
    xr = max_c + 1 + pad
    yb = -(max_r + 1 + pad)
    yt = -(min_r - pad)
    ax.set_xlim(xl, xr)
    ax.set_ylim(yb, yt)
    ax.set_aspect("equal")
    ax.set_facecolor(BG_CLR)

    # ── x=0 and y=0 reference lines ──────────────────────────────
    ax.axvline(x=0, color="#555", lw=2.0, ls="--", zorder=0.5, alpha=0.75)
    ax.axhline(y=0, color="#555", lw=2.0, ls="--", zorder=0.5, alpha=0.75)
    ax.text(0.12, yt - 0.15, "x = 0", fontsize=11, color="#444",
            alpha=0.85, fontfamily="monospace", va="top", fontweight="bold")
    ax.text(xl + 0.15, 0.12, "y = 0", fontsize=11, color="#444",
            alpha=0.85, fontfamily="monospace", va="bottom", fontweight="bold")

    # ── empty-neighbour placeholders ──────────────────────────────
    occupied = set(rec.board.keys())
    shown: set = set()
    for r, c in occupied:
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nb = (r+dr, c+dc)
            if nb not in occupied and nb not in shown:
                shown.add(nb)
                ax.add_patch(Rectangle(
                    (nb[1], -(nb[0]+1)), 1, 1,
                    fc=EMPTY_CLR, ec="#ccc", lw=0.4, ls="--",
                    alpha=0.25, zorder=0))

    # ── tiles ─────────────────────────────────────────────────────
    for (r, c), t in rec.board.items():
        _draw_tile(ax, r, c, t, highlight=(r, c) == rec.last_placed)

    # ── meeples ───────────────────────────────────────────────────
    for pidx, pms in enumerate(rec.placed_meeples):
        for mp in pms:
            coord = mp.coordinate_with_side.coordinate
            _draw_meeple(ax, coord.row, coord.column,
                         mp.coordinate_with_side.side, pidx, mp.meeple_type)

    # ── title / scoreboard ────────────────────────────────────────
    ax.set_title(
        f"Turn {rec.turn}   |   "
        f"P0 (red): {rec.scores[0]}    "
        f"P1 (blue): {rec.scores[1]}    "
        f"Deck: {rec.deck_remaining}",
        fontsize=18, fontweight="bold", pad=12)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    handles = [
        mpatches.Patch(fc=GRASS_CLR, ec="none", label="Grass"),
        mpatches.Patch(fc=CITY_CLR, ec="none", label="City"),
        mpatches.Patch(fc=ROAD_CLR, ec=ROAD_EDGE_CLR, label="Road"),
        mpatches.Patch(fc=PLAYER_CLR[0], ec="none", label="Player 0"),
        mpatches.Patch(fc=PLAYER_CLR[1], ec="none", label="Player 1"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=13,
              framealpha=0.85, ncol=1, handlelength=1.2)

    # ── Action log in the lower axes ──────────────────────────────
    ax_log.set_xlim(0, 1)
    ax_log.set_ylim(0, 1)
    ax_log.axis("off")
    ax_log.set_facecolor("white")

    fig_w_in = fig.get_size_inches()[0]
    log_fs = max(10, min(16, fig_w_in * 1.1))

    MAX_LOG_LINES = 9
    has_warn = bool(rec.warnings)
    raw = list(rec.action_log)

    endgame = [l for l in raw if l.startswith("END-GAME:")]
    if len(endgame) > 4:
        non_eg = [l for l in raw if not l.startswith("END-GAME:")]
        p_counts: Dict[str, int] = {}
        for l in endgame:
            tag = l.split()[1]
            p_counts[tag] = p_counts.get(tag, 0) + 1
        summary = "END-GAME: " + ", ".join(
            f"{tag} {n} meeples removed" for tag, n in sorted(p_counts.items()))
        non_eg.append(summary)
        raw = non_eg

    log_lines = raw[:MAX_LOG_LINES]
    if has_warn:
        log_lines = raw[:MAX_LOG_LINES - 2]
        for w in rec.warnings[:2]:
            log_lines.append(f"⚠ {w}")
    if len(raw) > MAX_LOG_LINES and not has_warn:
        log_lines.append(f"  … {len(raw) - MAX_LOG_LINES} more lines")
    log_text = "\n".join(log_lines)

    color = "#c1121f" if has_warn else "#222"
    ax_log.text(0.02, 0.96, log_text, transform=ax_log.transAxes,
                fontsize=log_fs, fontfamily="monospace", va="top", ha="left",
                color=color, linespacing=1.3,
                bbox=dict(boxstyle="round,pad=0.4", fc="white",
                          ec="#bbb" if not has_warn else "#c1121f",
                          lw=0.8, alpha=0.95))


# ═══════════════════════════════════════════════════════════════════════
# Animation / frame export
# ═══════════════════════════════════════════════════════════════════════
def _make_figure(records):
    extent = _board_extent(records)
    min_r, max_r, min_c, max_c = extent
    board_w = max_c - min_c + 1 + 3  # tile columns + padding
    board_h = max_r - min_r + 1 + 3  # tile rows + padding
    SCALE = 0.85                      # inches per tile unit
    fig_w = max(8, board_w * SCALE)
    board_h_in = max(5, board_h * SCALE)
    log_h_in = board_h_in / 2         # log panel = half board height
    fig_h = board_h_in + log_h_in + 0.8  # +0.8 for title
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(2, 1, height_ratios=[2, 1], hspace=0.05,
                          left=0.03, right=0.97, top=0.94, bottom=0.02)
    ax_board = fig.add_subplot(gs[0])
    ax_log = fig.add_subplot(gs[1])
    return fig, ax_board, ax_log, extent


def create_animation(records: List[TurnRecord],
                     path: str = "game_animation.mp4",
                     fps: int = 2,
                     hold_last_secs: int = 2):
    fig, ax, ax_log, extent = _make_figure(records)

    hold_frames = hold_last_secs * fps
    total_frames = len(records) + hold_frames

    def _update(i):
        idx = min(i, len(records) - 1)
        render_frame(fig, ax, ax_log, records[idx], extent)

    anim = animation.FuncAnimation(
        fig, _update, frames=total_frames, interval=1000 / fps, blit=False)

    try:
        writer = animation.FFMpegWriter(fps=fps, bitrate=2000)
        anim.save(path, writer=writer)
        print(f"Saved animation  →  {path}")
    except Exception as exc:
        gif_path = path.rsplit(".", 1)[0] + ".gif"
        print(f"FFmpeg unavailable ({exc}), saving GIF instead …")
        anim.save(gif_path, writer="pillow", fps=fps)
        print(f"Saved animation  →  {gif_path}")
    plt.close(fig)


def save_frames(records: List[TurnRecord], out_dir: str = "game_frames"):
    os.makedirs(out_dir, exist_ok=True)
    extent = _board_extent(records)

    for rec in records:
        fig, ax, ax_log, _ = _make_figure(records)
        render_frame(fig, ax, ax_log, rec, extent)
        fname = os.path.join(out_dir, f"turn_{rec.turn:03d}.png")
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
    print(f"Saved {len(records)} frames  →  {out_dir}/")


# ═══════════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(
        description="Play a random Carcassonne game and produce an animation.")
    ap.add_argument("--seed", type=int, default=None,
                    help="Random seed for reproducibility")
    ap.add_argument("--fps", type=int, default=2,
                    help="Frames per second in the animation (default: 2)")
    ap.add_argument("--output", default="game_animation.mp4",
                    help="Output file path (default: game_animation.mp4)")
    ap.add_argument("--frames", action="store_true",
                    help="Also save individual turn PNGs to game_frames/")
    ap.add_argument("--max-turns", type=int, default=None,
                    help="Stop after N turns (for quick testing)")
    ap.add_argument("--variant", choices=list(_VARIANTS.keys()), default="full",
                    help="Game variant: 'full' (I&C + abbots) or 'base' (default: full)")
    args = ap.parse_args()

    print(f"Playing random game (variant={args.variant}) …")
    records = play_random_game(seed=args.seed, max_turns=args.max_turns,
                               variant=args.variant)
    if not records:
        print("No turns were played – something went wrong.")
        return

    n_warn = sum(len(r.warnings) for r in records)
    print(f"Game over — {len(records)} turns.  "
          f"Final score: P0={records[-1].scores[0]}  "
          f"P1={records[-1].scores[1]}")
    if n_warn:
        print(f"  ⚠ {n_warn} meeple-removal warning(s) detected!")
    else:
        print(f"  ✓ All meeple removals verified correct.")

    if args.frames:
        save_frames(records)

    print("Rendering animation …")
    create_animation(records, path=args.output, fps=args.fps)
    print("Done.")


if __name__ == "__main__":
    main()

"""Game statistics tracking for Carcassonne self-play and arena games.

Tracks per-player action counts (tiles placed, meeples placed by type,
passes), per-player score breakdown (mid-game / abbot / final-scoring),
and a board-level feature breakdown at the terminal state.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

try:
    from carcassonne_engine import (
        MeepleAction, PassAction, TileAction, MeepleType, GamePhase,
        CityUtil, RoadUtil, PointsCollector,
        Coordinate,
    )
except ImportError:
    from carcassonne.objects.actions.meeple_action import MeepleAction
    from carcassonne.objects.actions.pass_action import PassAction
    from carcassonne.objects.actions.tile_action import TileAction
    from carcassonne.objects.meeple_type import MeepleType
    from carcassonne.objects.game_phase import GamePhase
    from carcassonne.utils.city_util import CityUtil
    from carcassonne.utils.road_util import RoadUtil
    from carcassonne.utils.points_collector import PointsCollector
    from carcassonne.objects.coordinate import Coordinate


@dataclass
class PlayerStats:
    tiles_placed: int = 0
    meeples_placed: int = 0
    normal_meeples: int = 0
    farmers: int = 0
    big_meeples: int = 0
    big_farmers: int = 0
    abbots: int = 0
    meeple_removals: int = 0
    passes: int = 0
    final_score: int = 0
    mid_game_pts: int = 0
    abbot_pts: int = 0
    final_scoring_pts: int = 0


@dataclass
class BoardFeatures:
    """Board-level feature counts and point totals (not per-player)."""
    completed_cities: int = 0
    completed_city_pts: int = 0
    incomplete_cities: int = 0
    incomplete_city_pts: int = 0
    completed_roads: int = 0
    completed_road_pts: int = 0
    incomplete_roads: int = 0
    incomplete_road_pts: int = 0
    chapels_flowers: int = 0
    chapels_flowers_pts: int = 0

    def summary(self) -> str:
        return (
            f"cities(done={self.completed_cities}/{self.completed_city_pts}pts, "
            f"open={self.incomplete_cities}/{self.incomplete_city_pts}pts) "
            f"roads(done={self.completed_roads}/{self.completed_road_pts}pts, "
            f"open={self.incomplete_roads}/{self.incomplete_road_pts}pts) "
            f"chapels({self.chapels_flowers}/{self.chapels_flowers_pts}pts)"
        )


@dataclass
class GameStats:
    """Accumulates per-player statistics during a game."""
    num_players: int = 2
    players: List[PlayerStats] = field(default_factory=list)
    board_features: Optional[BoardFeatures] = None
    total_turns: int = 0
    _prev_scores: List[int] = field(default_factory=list, repr=False)
    _last_action: Any = field(default=None, repr=False)
    _last_phase: Any = field(default=None, repr=False)

    def __post_init__(self):
        if not self.players:
            self.players = [PlayerStats() for _ in range(self.num_players)]
        if not self._prev_scores:
            self._prev_scores = [0] * self.num_players

    def record_action(self, player: int, action: Any, pre_state: Any = None) -> None:
        """Record action type counts and snapshot pre-action phase.

        Call this BEFORE apply_action. Pass *pre_state* (the state before the
        action is applied) so we can read the current game phase.
        """
        ps = self.players[player]
        self.total_turns += 1
        self._last_action = action
        if pre_state is not None:
            self._last_phase = pre_state.lib_state.phase
        if isinstance(action, TileAction):
            ps.tiles_placed += 1
        elif isinstance(action, MeepleAction):
            if action.remove:
                ps.meeple_removals += 1
            else:
                ps.meeples_placed += 1
                mt = action.meeple_type
                if mt == MeepleType.NORMAL:
                    ps.normal_meeples += 1
                elif mt == MeepleType.FARMER:
                    ps.farmers += 1
                elif mt == MeepleType.BIG:
                    ps.big_meeples += 1
                elif mt == MeepleType.BIG_FARMER:
                    ps.big_farmers += 1
                elif mt == MeepleType.ABBOT:
                    ps.abbots += 1
        elif isinstance(action, PassAction):
            ps.passes += 1

    def record_score_delta(self, new_state) -> None:
        """Compute per-player score deltas and classify them.

        Call this AFTER apply_action with the resulting new_state.
        Classification:
          - Terminal state delta → final_scoring_pts
          - Abbot removal delta → abbot_pts
          - Otherwise (meeple-phase resolution) → mid_game_pts
        """
        is_terminal = new_state.is_terminal()
        for i in range(self.num_players):
            cur = int(new_state.get_player_score(i))
            delta = cur - self._prev_scores[i]
            self._prev_scores[i] = cur
            if delta == 0:
                continue
            if is_terminal:
                self.players[i].final_scoring_pts += delta
            elif (isinstance(self._last_action, MeepleAction)
                  and self._last_action.remove
                  and self._last_action.meeple_type == MeepleType.ABBOT):
                self.players[i].abbot_pts += delta
            else:
                self.players[i].mid_game_pts += delta

    def set_final_scores(self, state) -> None:
        for i in range(self.num_players):
            self.players[i].final_score = int(state.get_player_score(i))
        try:
            self.board_features = _scan_board_features(state)
        except Exception:
            self.board_features = None

    def summary_line(self, prefix: str = "") -> str:
        parts = [prefix] if prefix else []
        for i, ps in enumerate(self.players):
            parts.append(
                f"P{i}(score={ps.final_score}"
                f"[mid={ps.mid_game_pts},final={ps.final_scoring_pts},"
                f"abbot={ps.abbot_pts}], "
                f"meeples={ps.meeples_placed}["
                f"N={ps.normal_meeples},F={ps.farmers},"
                f"B={ps.big_meeples},BF={ps.big_farmers},"
                f"A={ps.abbots}], "
                f"tiles={ps.tiles_placed}, "
                f"passes={ps.passes})"
            )
        total_meeples = sum(ps.meeples_placed for ps in self.players)
        parts.append(f"total_meeples={total_meeples}, turns={self.total_turns}")
        line = " | ".join(parts)
        if self.board_features:
            line += f"\n  Board: {self.board_features.summary()}"
        return line

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "total_turns": self.total_turns,
            "players": [
                {
                    "score": ps.final_score,
                    "mid_game_pts": ps.mid_game_pts,
                    "final_scoring_pts": ps.final_scoring_pts,
                    "abbot_pts": ps.abbot_pts,
                    "tiles_placed": ps.tiles_placed,
                    "meeples_placed": ps.meeples_placed,
                    "normal_meeples": ps.normal_meeples,
                    "farmers": ps.farmers,
                    "big_meeples": ps.big_meeples,
                    "big_farmers": ps.big_farmers,
                    "abbots": ps.abbots,
                    "meeple_removals": ps.meeple_removals,
                    "passes": ps.passes,
                }
                for ps in self.players
            ],
        }
        if self.board_features:
            bf = self.board_features
            d["board_features"] = {
                "completed_cities": bf.completed_cities,
                "completed_city_pts": bf.completed_city_pts,
                "incomplete_cities": bf.incomplete_cities,
                "incomplete_city_pts": bf.incomplete_city_pts,
                "completed_roads": bf.completed_roads,
                "completed_road_pts": bf.completed_road_pts,
                "incomplete_roads": bf.incomplete_roads,
                "incomplete_road_pts": bf.incomplete_road_pts,
                "chapels_flowers": bf.chapels_flowers,
                "chapels_flowers_pts": bf.chapels_flowers_pts,
            }
        return d


def _scan_board_features(state) -> BoardFeatures:
    """Enumerate all cities, roads, and chapels on the terminal board."""
    gs = state.lib_state
    board = gs.board
    bf = BoardFeatures()

    all_coords = [Coordinate(row=r, column=c) for (r, c) in board.keys()]

    seen_cities = set()
    seen_roads = set()

    for coord in all_coords:
        for city in CityUtil.find_cities(gs, coord):
            if city in seen_cities:
                continue
            seen_cities.add(city)
            pts = PointsCollector.count_city_points(gs, city)
            if city.finished:
                bf.completed_cities += 1
                bf.completed_city_pts += pts
            else:
                bf.incomplete_cities += 1
                bf.incomplete_city_pts += pts

        for road in RoadUtil.find_roads(gs, coord):
            if road in seen_roads:
                continue
            seen_roads.add(road)
            pts = PointsCollector.count_road_points(gs, road)
            if road.finished:
                bf.completed_roads += 1
                bf.completed_road_pts += pts
            else:
                bf.incomplete_roads += 1
                bf.incomplete_road_pts += pts

        tile = gs.get_tile(coord.row, coord.column)
        if tile is not None and (tile.chapel or tile.flowers):
            pts = PointsCollector.chapel_or_flowers_points(gs, coord)
            bf.chapels_flowers += 1
            bf.chapels_flowers_pts += pts

    return bf

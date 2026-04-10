try:
    from carcassonne_engine import (
        Coordinate, SupplementaryRule, TileSet,
        CarcassonneGameState as LibCarcassonneGameState,
    )
except ImportError:
    from carcassonne.objects.coordinate import Coordinate
    from carcassonne.tile_sets.supplementary_rules import SupplementaryRule
    from carcassonne.tile_sets.tile_sets import TileSet
    from carcassonne.carcassonne_game_state import CarcassonneGameState as LibCarcassonneGameState

from .carcassonne_game_state import CarcassonneGameState
from .game_state import GameState
from .game import Game

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


class CarcassonneGame(Game[CarcassonneGameState]):
    def __init__(self, variant: str = "full") -> None:
        super().__init__()
        if variant not in _VARIANTS:
            raise ValueError(f"Unknown game variant '{variant}'. "
                             f"Choose from: {list(_VARIANTS.keys())}")
        self.variant = variant
        cfg = _VARIANTS[variant]
        self._tile_sets = cfg["tile_sets"]
        self._supplementary_rules = cfg["supplementary_rules"]

    def get_initial_state(self) -> GameState:
        players = 2
        if TileSet.THE_RIVER in self._tile_sets:
            raise NotImplementedError(
                "The river has bug related to get_river_rotation_ends "
                "(need to find the right starting end for previous_river_ends). "
                "Do not use")

        self.supplementary_rules = self._supplementary_rules
        lib_state = LibCarcassonneGameState(
            tile_sets=self._tile_sets,
            players=players,
            supplementary_rules=self._supplementary_rules,
            starting_position=Coordinate(0, 0),
        )
        return CarcassonneGameState(lib_state)

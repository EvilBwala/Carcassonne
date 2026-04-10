"""
Carcassonne game engine - compatibility layer.

Tries to load the C++ extension first; falls back to the pure Python engine.
"""

try:
    from _carcassonne_engine import (
        # Enums & constants
        GamePhase,
        TerrainType,
        Rotation,
        SupplementaryRule,
        TileSet,
        # Types
        Coordinate,
        CoordinateWithSide,
        CoordinateWithFarmerSide,
        Connection,
        FarmerConnection,
        FarmerConnectionWithCoordinate,
        Tile,
        MeeplePosition,
        City,
        Road,
        Farm,
        PlayingPosition,
        # Actions
        TileAction,
        MeepleAction,
        PassAction,
        # Game state
        CarcassonneGameState,
        TileRegistry,
        tile_registry,
        # Utilities
        ActionUtil,
        StateUpdater,
        TileFitter,
        CityUtil,
        RoadUtil,
        FarmUtil,
        PointsCollector,
        # Functions
        action_to_json,
        action_from_json,
        # Tile data
        base_tiles,
        inns_and_cathedrals_tiles,
    )
    # Submodules for constants
    from _carcassonne_engine import Side
    from _carcassonne_engine import MeepleType
    from _carcassonne_engine import FarmerSide
    USING_CPP_ENGINE = True

except ImportError:
    from carcassonne.carcassonne_game_state import (
        CarcassonneGameState,
        TileRegistry,
        action_from_json,
        action_to_json,
        tile_registry,
    )
    from carcassonne.objects.actions.action import Action
    from carcassonne.objects.actions.tile_action import TileAction
    from carcassonne.objects.actions.meeple_action import MeepleAction
    from carcassonne.objects.actions.pass_action import PassAction
    from carcassonne.objects.coordinate import Coordinate
    from carcassonne.objects.coordinate_with_side import CoordinateWithSide
    from carcassonne.objects.coordinate_with_farmer_side import CoordinateWithFarmerSide
    from carcassonne.objects.connection import Connection
    from carcassonne.objects.farmer_connection import FarmerConnection
    from carcassonne.objects.farmer_connection_with_coordinate import FarmerConnectionWithCoordinate
    from carcassonne.objects.tile import Tile
    from carcassonne.objects.meeple_position import MeeplePosition
    from carcassonne.objects.city import City
    from carcassonne.objects.road import Road
    from carcassonne.objects.farm import Farm
    from carcassonne.objects.playing_position import PlayingPosition
    from carcassonne.objects.game_phase import GamePhase
    from carcassonne.objects.terrain_type import TerrainType
    from carcassonne.objects.rotation import Rotation
    from carcassonne.objects.side import Side
    from carcassonne.objects.meeple_type import MeepleType
    from carcassonne.objects.farmer_side import FarmerSide
    from carcassonne.tile_sets.supplementary_rules import SupplementaryRule
    from carcassonne.tile_sets.tile_sets import TileSet
    from carcassonne.tile_sets.base_deck import base_tiles
    from carcassonne.tile_sets.inns_and_cathedrals_deck import inns_and_cathedrals_tiles
    from carcassonne.utils.action_util import ActionUtil
    from carcassonne.utils.state_updater import StateUpdater
    from carcassonne.utils.tile_fitter import TileFitter
    from carcassonne.utils.city_util import CityUtil
    from carcassonne.utils.road_util import RoadUtil
    from carcassonne.utils.farm_util import FarmUtil
    from carcassonne.utils.points_collector import PointsCollector

    USING_CPP_ENGINE = False

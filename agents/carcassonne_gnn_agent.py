"""
Graph Neural Network / Graph Transformer agent for Carcassonne AlphaZero.

Two model variants live here:
  1. GNNCarcassonneModel   — legacy 4-layer GATv2Conv backbone
  2. GPSCarcassonneModel   — GPS (General, Powerful, Scalable) Graph Transformer
     with rotation/translation/chirality invariant features:
       • Canonical terrain sequence (20-dim, rotation-invariant, chirality-preserving)
       • Random Walk Positional Encoding (RWPE, 32-dim, topology-only)
       • 3-dim terrain edge features (no direction bits fed to model)
       • 8 GPS layers: local GATv2Conv + global multi-head self-attention

Board state → PyG Data → encode → value head + per-action MLP scorer.
"""

from __future__ import annotations

import math
import os
import random
from collections import deque
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm

from torch_geometric.data import Batch, Data
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch_geometric.utils import softmax as pyg_softmax

try:
    from carcassonne_engine import (
        MeepleAction, PassAction, TileAction, Coordinate, GamePhase,
        TerrainType, Tile, base_tiles, inns_and_cathedrals_tiles, TileFitter,
    )
    from carcassonne_engine import Side
except ImportError:
    from carcassonne.objects.actions.meeple_action import MeepleAction
    from carcassonne.objects.actions.pass_action import PassAction
    from carcassonne.objects.actions.tile_action import TileAction
    from carcassonne.objects.coordinate import Coordinate
    from carcassonne.objects.game_phase import GamePhase
    from carcassonne.objects.side import Side
    from carcassonne.objects.terrain_type import TerrainType
    from carcassonne.objects.tile import Tile
    from carcassonne.tile_sets.base_deck import base_tiles
    from carcassonne.tile_sets.inns_and_cathedrals_deck import inns_and_cathedrals_tiles
    from carcassonne.utils.tile_fitter import TileFitter

try:
    import _carcassonne_engine as _cpp_engine
    _HAS_CPP_GRAPH_BUILD = hasattr(_cpp_engine, 'build_graph_features')
except ImportError:
    _HAS_CPP_GRAPH_BUILD = False

from games.carcassonne_game_state import CarcassonneGameState
from utils.utils import ModelSummarizer, normalize_dict, save_model

from .mcts import MCTS, MCTSConfig
from .mcts_ml_agent import MCTSMLAgent, MCTSMLAgentConfig

# ─────────────────────────────────────────────────────────────────────────────
# 1. Tile Vocabulary  (base + Inns & Cathedrals — 50 unique types)
# ─────────────────────────────────────────────────────────────────────────────

TILE_DESCRIPTIONS: List[str] = sorted(
    list(base_tiles.keys()) + list(inns_and_cathedrals_tiles.keys())
)
TILE_DESC_TO_IDX: Dict[str, int] = {d: i for i, d in enumerate(TILE_DESCRIPTIONS)}
NUM_TILE_TYPES: int = len(TILE_DESCRIPTIONS)   # 50

# ─────────────────────────────────────────────────────────────────────────────
# 2. Feature-dimension constants
# ─────────────────────────────────────────────────────────────────────────────

NUM_ROTATIONS      = 4
NUM_TERRAIN_TYPES  = 5   # CITY / ROAD / GRASS / CHAPEL / OTHER
NUM_CARDINAL       = 4   # TOP / RIGHT / BOTTOM / LEFT
NUM_MEEPLE_TYPES   = 5   # NORMAL / ABBOT / FARMER / BIG / BIG_FARMER
NUM_SIDES          = 9   # TOP=0 … CENTER=8
NUM_TILE_FLAGS     = 4   # chapel, shield, cathedral, flowers
NUM_GAME_GLOBALS   = 7   # cur/enemy resources + is_meeples_phase

TERRAIN_PER_SIDE_DIM = NUM_CARDINAL * NUM_TERRAIN_TYPES  # 20
MEEPLE_OWNER_DIM     = 3                                 # none / player-0 / player-1

# ── Legacy (rotation-VARIANT) tile static features ───────────────────────
TILE_STATIC_DIM = (
    NUM_TILE_TYPES        # 50
    + NUM_ROTATIONS       # 4
    + TERRAIN_PER_SIDE_DIM  # 20
    + NUM_TILE_FLAGS      # 4
)  # = 78

NODE_FEAT_DIM = (
    TILE_STATIC_DIM       # 78
    + 1                   # just_placed
    + 1                   # has_meeple
    + MEEPLE_OWNER_DIM    # 3
    + NUM_MEEPLE_TYPES    # 5
    + NUM_SIDES           # 9
    + NUM_GAME_GLOBALS    # 7
)  # = 104

EDGE_FEAT_DIM = 7   # CITY / ROAD / GRASS / d_TOP / d_RIGHT / d_BOTTOM / d_LEFT

GNN_HIDDEN           = 128
TILE_ACTION_FEAT_DIM  = TILE_STATIC_DIM + GNN_HIDDEN + EDGE_FEAT_DIM   # 213
MEEPLE_ACTION_FEAT_DIM = GNN_HIDDEN + NUM_MEEPLE_TYPES + NUM_SIDES      # 142

# ── GPS (rotation-INVARIANT) feature dimensions ─────────────────────────
CANON_TERRAIN_DIM = TERRAIN_PER_SIDE_DIM                     # 20
RWPE_DIM          = 32
GPS_LAYERS        = 8
GPS_HEADS         = 4
EDGE_MODEL_DIM    = 3   # terrain-only edge features for the model (CITY/ROAD/GRASS)

GPS_TILE_STATIC_DIM = (
    NUM_TILE_TYPES        # 50  (rotation-invariant: description doesn't change with rotation)
    + CANON_TERRAIN_DIM   # 20  (rotation-invariant canonical terrain sequence)
    + NUM_TILE_FLAGS      # 4
)  # = 74

GPS_NODE_FEAT_DIM = (
    GPS_TILE_STATIC_DIM   # 74
    + 1                   # just_placed
    + 1                   # has_meeple
    + MEEPLE_OWNER_DIM    # 3
    + NUM_MEEPLE_TYPES    # 5
    + NUM_SIDES           # 9
    + NUM_GAME_GLOBALS    # 7
)  # = 100   (RWPE is added separately via a learned encoder)

GPS_TILE_ACTION_FEAT_DIM  = GPS_TILE_STATIC_DIM + GNN_HIDDEN + EDGE_MODEL_DIM  # 205
GPS_MEEPLE_ACTION_FEAT_DIM = GNN_HIDDEN + NUM_MEEPLE_TYPES + NUM_SIDES          # 142

MODEL_SIZE_PRESETS: Dict[str, Dict[str, int]] = {
    'default': {'gps_layers': 8, 'gnn_hidden': 128, 'gps_heads': 4},
    'small':   {'gps_layers': 4, 'gnn_hidden': 64,  'gps_heads': 2},
}

# Number of local distance categories for attention bias:
#   0 = self, 1 = cardinal neighbour (d=1), 2 = diagonal neighbour (d=√2),
#   3 = two-step cardinal neighbour (d=2)
NUM_DIST_CATEGORIES = 4

# ─────────────────────────────────────────────────────────────────────────────
# 3. Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────

CARDINAL: List[int] = [Side.TOP, Side.RIGHT, Side.BOTTOM, Side.LEFT]

DIR_DELTAS: Dict[int, Tuple[int, int]] = {
    Side.TOP:    (-1,  0),
    Side.RIGHT:  ( 0,  1),
    Side.BOTTOM: ( 1,  0),
    Side.LEFT:   ( 0, -1),
}

_TERRAIN_IDX: Dict[TerrainType, int] = {
    TerrainType.CITY:    0,
    TerrainType.ROAD:    1,
    TerrainType.GRASS:   2,
    TerrainType.CHAPEL:  3,
    TerrainType.FLOWERS: 3,
}
_EDGE_TERRAIN_IDX: Dict[TerrainType, int] = {
    TerrainType.CITY:  0,
    TerrainType.ROAD:  1,
    TerrainType.GRASS: 2,
}

_DIRECTION_IDX: Dict[int, int] = {
    Side.TOP:    0,
    Side.RIGHT:  1,
    Side.BOTTOM: 2,
    Side.LEFT:   3,
}
_IDX_TO_DIRECTION: List[int] = [Side.TOP, Side.RIGHT, Side.BOTTOM, Side.LEFT]

OPPOSITE_DIR: Dict[int, int] = {
    Side.TOP:    Side.BOTTOM,
    Side.BOTTOM: Side.TOP,
    Side.LEFT:   Side.RIGHT,
    Side.RIGHT:  Side.LEFT,
}

# ─────────────────────────────────────────────────────────────────────────────
# 4. Encoding helpers
# ─────────────────────────────────────────────────────────────────────────────

def _terrain_one_hot(terrain: Optional[TerrainType]) -> np.ndarray:
    """5-dim one-hot (CITY / ROAD / GRASS / CHAPEL+FLOWERS / OTHER)."""
    v = np.zeros(NUM_TERRAIN_TYPES, dtype=np.float32)
    if terrain is not None:
        idx = _TERRAIN_IDX.get(terrain, NUM_TERRAIN_TYPES - 1)
        v[idx] = 1.0
    return v


def canonical_terrain_sequence(tile: Tile) -> np.ndarray:
    """
    20-dim rotation-invariant, chirality-preserving terrain feature.

    Extracts the terrain one-hot (5-dim) for each cardinal side in clockwise
    order [TOP, RIGHT, BOTTOM, LEFT], giving a 20-dim vector.  Considers all
    4 cyclic rotations of this sequence and picks the lexicographically
    smallest.  Mirror reflections are NOT considered, so chiral tile pairs
    (e.g. city-road-grass-grass vs city-grass-grass-road) produce different
    canonical forms.
    """
    sides_oh: List[np.ndarray] = []
    for side in CARDINAL:
        sides_oh.append(_terrain_one_hot(tile.get_type(side)))

    candidates: List[np.ndarray] = []
    for shift in range(4):
        rotated = np.concatenate([sides_oh[(shift + i) % 4] for i in range(4)])
        candidates.append(rotated)

    best = candidates[0]
    for c in candidates[1:]:
        if tuple(c) < tuple(best):
            best = c
    return best


def encode_tile_static(tile: Tile) -> np.ndarray:
    """
    78-dim tile-only feature: type one-hot + rotation + terrain-per-side + flags.
    Stateless w.r.t. meeples and game context.  (Legacy, NOT rotation-invariant.)
    """
    buf = np.zeros(TILE_STATIC_DIM, dtype=np.float32)
    off = 0

    idx = TILE_DESC_TO_IDX.get(tile.description, -1)
    if idx >= 0:
        buf[off + idx] = 1.0
    off += NUM_TILE_TYPES

    buf[off + min(tile.turns, NUM_ROTATIONS - 1)] = 1.0
    off += NUM_ROTATIONS

    for side in CARDINAL:
        buf[off: off + NUM_TERRAIN_TYPES] = _terrain_one_hot(tile.get_type(side))
        off += NUM_TERRAIN_TYPES

    buf[off + 0] = float(tile.chapel)
    buf[off + 1] = float(tile.shield)
    buf[off + 2] = float(tile.cathedral)
    buf[off + 3] = float(tile.flowers)

    return buf


def encode_gps_tile_static(tile: Tile) -> np.ndarray:
    """
    74-dim rotation-invariant tile-only feature:
      [0:50]  tile-type one-hot (rotation-invariant: description is unchanged by turn())
      [50:70] canonical terrain sequence (rotation-invariant, chirality-preserving)
      [70:74] flags (chapel, shield, cathedral, flowers)

    The tile-type one-hot captures internal connectivity (e.g. connected vs
    disconnected cities, straight roads vs crossroads) that the canonical
    terrain sequence alone cannot distinguish.
    """
    buf = np.zeros(GPS_TILE_STATIC_DIM, dtype=np.float32)
    off = 0

    idx = TILE_DESC_TO_IDX.get(tile.description, -1)
    if idx >= 0:
        buf[off + idx] = 1.0
    off += NUM_TILE_TYPES

    buf[off: off + CANON_TERRAIN_DIM] = canonical_terrain_sequence(tile)
    off += CANON_TERRAIN_DIM

    buf[off + 0] = float(tile.chapel)
    buf[off + 1] = float(tile.shield)
    buf[off + 2] = float(tile.cathedral)
    buf[off + 3] = float(tile.flowers)

    return buf


def encode_tile_node_features(
    tile: Tile,
    is_just_placed: bool,
    meeple_owner: int,
    meeple_type: int,
    meeple_side: int,
    player_meeples: int,
    player_abbots: int,
    player_big_meeples: int,
    enemy_meeples: int,
    enemy_abbots: int,
    enemy_big_meeples: int,
    is_meeples_phase: bool,
) -> np.ndarray:
    """104-dim node feature vector (legacy, NOT rotation-invariant)."""
    buf = np.zeros(NODE_FEAT_DIM, dtype=np.float32)
    off = 0

    buf[off: off + TILE_STATIC_DIM] = encode_tile_static(tile)
    off += TILE_STATIC_DIM

    buf[off] = float(is_just_placed);            off += 1
    buf[off] = float(meeple_owner >= 0);         off += 1

    if meeple_owner == 0:   buf[off + 1] = 1.0
    elif meeple_owner == 1: buf[off + 2] = 1.0
    else:                   buf[off + 0] = 1.0
    off += MEEPLE_OWNER_DIM

    if 0 <= meeple_type < NUM_MEEPLE_TYPES:
        buf[off + meeple_type] = 1.0
    off += NUM_MEEPLE_TYPES

    if 0 <= meeple_side < NUM_SIDES:
        buf[off + meeple_side] = 1.0
    off += NUM_SIDES

    buf[off + 0] = player_meeples / 7.0
    buf[off + 1] = float(player_abbots)
    buf[off + 2] = float(player_big_meeples)
    buf[off + 3] = enemy_meeples / 7.0
    buf[off + 4] = float(enemy_abbots)
    buf[off + 5] = float(enemy_big_meeples)
    buf[off + 6] = float(is_meeples_phase)
    return buf


def encode_gps_node_features(
    tile: Tile,
    is_just_placed: bool,
    meeple_owner: int,
    meeple_type: int,
    meeple_side: int,
    player_meeples: int,
    player_abbots: int,
    player_big_meeples: int,
    enemy_meeples: int,
    enemy_abbots: int,
    enemy_big_meeples: int,
    is_meeples_phase: bool,
) -> np.ndarray:
    """
    100-dim rotation-invariant node feature vector (GPS model).
      [0:74]   GPS tile static (type one-hot + canonical terrain + flags)
      [74]     just_placed
      [75]     has_meeple
      [76:79]  meeple_owner one-hot
      [79:84]  meeple_type one-hot
      [84:93]  meeple_side one-hot
      [93:100] game globals
    RWPE (32-dim) is stored separately and injected via a learned encoder.
    """
    buf = np.zeros(GPS_NODE_FEAT_DIM, dtype=np.float32)
    off = 0

    buf[off: off + GPS_TILE_STATIC_DIM] = encode_gps_tile_static(tile)
    off += GPS_TILE_STATIC_DIM

    buf[off] = float(is_just_placed);            off += 1
    buf[off] = float(meeple_owner >= 0);         off += 1

    if meeple_owner == 0:   buf[off + 1] = 1.0
    elif meeple_owner == 1: buf[off + 2] = 1.0
    else:                   buf[off + 0] = 1.0
    off += MEEPLE_OWNER_DIM

    if 0 <= meeple_type < NUM_MEEPLE_TYPES:
        buf[off + meeple_type] = 1.0
    off += NUM_MEEPLE_TYPES

    if 0 <= meeple_side < NUM_SIDES:
        buf[off + meeple_side] = 1.0
    off += NUM_SIDES

    buf[off + 0] = player_meeples / 7.0
    buf[off + 1] = float(player_abbots)
    buf[off + 2] = float(player_big_meeples)
    buf[off + 3] = enemy_meeples / 7.0
    buf[off + 4] = float(enemy_abbots)
    buf[off + 5] = float(enemy_big_meeples)
    buf[off + 6] = float(is_meeples_phase)
    return buf


def get_edge_terrain(tile: Tile, direction: int) -> np.ndarray:
    """
    7-dim edge feature: terrain one-hot (CITY/ROAD/GRASS) + 4-dim direction one-hot.
    The direction bits are metadata for BFS; not fed to the GPS model.
    """
    terrain = tile.get_type(direction)
    v = np.zeros(EDGE_FEAT_DIM, dtype=np.float32)
    terrain_idx = _EDGE_TERRAIN_IDX.get(terrain, -1)
    if terrain_idx >= 0:
        v[terrain_idx] = 1.0
    dir_idx = _DIRECTION_IDX.get(direction, -1)
    if dir_idx >= 0:
        v[3 + dir_idx] = 1.0
    return v


def compute_rwpe(edge_index: Tensor, num_nodes: int, k: int = RWPE_DIM) -> Tensor:
    """
    Random Walk Positional Encoding.

    For each node i, returns [RW^1_{ii}, RW^2_{ii}, ..., RW^K_{ii}] where
    RW = AD^{-1} is the random walk operator (column-stochastic).
    These diagonal entries capture the probability of returning to node i
    after 1..K steps.  Depends only on graph topology — invariant to any
    spatial transformation (translation, rotation).
    """
    if num_nodes == 0:
        return torch.zeros((0, k), dtype=torch.float32)

    device = edge_index.device if edge_index.numel() > 0 else torch.device('cpu')
    N = num_nodes

    if edge_index.numel() == 0:
        return torch.zeros((N, k), dtype=torch.float32, device=device)

    src, dst = edge_index[0], edge_index[1]

    # Degree of each node (number of outgoing edges)
    deg = torch.zeros(N, dtype=torch.float32, device=device)
    deg.scatter_add_(0, src, torch.ones_like(src, dtype=torch.float32))
    deg_inv = torch.zeros_like(deg)
    mask = deg > 0
    deg_inv[mask] = 1.0 / deg[mask]

    # Transition matrix T: T[dst, src] = 1/deg(src)   (column-stochastic)
    weights = deg_inv[src]

    pe = torch.zeros((N, k), dtype=torch.float32, device=device)

    if N <= 500:
        T = torch.zeros((N, N), dtype=torch.float32, device=device)
        T[dst, src] = weights
        Tk = torch.eye(N, dtype=torch.float32, device=device)
        for step in range(k):
            Tk = T @ Tk
            pe[:, step] = Tk.diagonal()
    else:
        T_sparse = torch.sparse_coo_tensor(
            torch.stack([dst, src]), weights, size=(N, N), device=device
        ).coalesce()
        Tk = torch.eye(N, dtype=torch.float32, device=device)
        for step in range(k):
            Tk = torch.sparse.mm(T_sparse, Tk)
            pe[:, step] = Tk.diagonal()

    return pe


def compute_local_distance_categories(
    pos_to_idx: Dict[Tuple[int, int], int],
    N: int,
) -> Tensor:
    """Compute a sparse local distance category tensor for attention bias.

    For every ordered pair (i, j) within 3 shells of the grid, assigns a
    category:  0 = self,  1 = cardinal (d=1),  2 = diagonal (d=√2),
               3 = two-step cardinal (d=2).
    Pairs beyond these shells are not stored (treated as "no bias" = 0.0).

    Returns
    -------
    dist_cats : LongTensor of shape (2, K) where K = number of local pairs.
        Row 0 = category (0..3), row 1 = flat index i*N+j for each pair.
        This compact representation avoids materialising a dense (N, N) matrix.
    """
    SHELL_OFFSETS: List[Tuple[int, int, int]] = [
        # (dr, dc, category)
        ( 0,  0, 0),  # self
        (-1,  0, 1), ( 1,  0, 1), ( 0, -1, 1), ( 0,  1, 1),  # cardinal
        (-1, -1, 2), (-1,  1, 2), ( 1, -1, 2), ( 1,  1, 2),  # diagonal
        (-2,  0, 3), ( 2,  0, 3), ( 0, -2, 3), ( 0,  2, 3),  # two-step cardinal
    ]

    cats: List[int] = []
    flat_indices: List[int] = []

    for (r, c), idx_i in pos_to_idx.items():
        for dr, dc, cat in SHELL_OFFSETS:
            idx_j = pos_to_idx.get((r + dr, c + dc), -1)
            if idx_j < 0:
                continue
            cats.append(cat)
            flat_indices.append(idx_i * N + idx_j)

    if not cats:
        return torch.zeros((2, 0), dtype=torch.long)

    return torch.stack([
        torch.tensor(cats, dtype=torch.long),
        torch.tensor(flat_indices, dtype=torch.long),
    ], dim=0)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Graph construction
# ─────────────────────────────────────────────────────────────────────────────

def _build_graph_from_state_cpp(state: CarcassonneGameState) -> Data:
    """Fast C++ path for GPS graph construction. RWPE still computed in Python."""
    raw = _cpp_engine.build_graph_features(state.lib_state, TILE_DESC_TO_IDX, NUM_TILE_TYPES)
    N = raw['num_nodes']
    E = raw['num_edges']

    if N > 0:
        x_gps_np = np.array(raw['x_gps'], dtype=np.float32).reshape(N, GPS_NODE_FEAT_DIM)
        x_gps = torch.from_numpy(x_gps_np)
    else:
        x_gps = torch.zeros((0, GPS_NODE_FEAT_DIM), dtype=torch.float32)

    if E > 0:
        ei_np = np.array(raw['edge_index'], dtype=np.int64).reshape(2, E)
        edge_index = torch.from_numpy(ei_np)
        ea_np = np.array(raw['edge_attr'], dtype=np.float32).reshape(E, EDGE_FEAT_DIM)
        edge_attr = torch.from_numpy(ea_np)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, EDGE_FEAT_DIM), dtype=torch.float32)

    # Legacy x (unused by GPS but needed for Data shape)
    x_tensor = torch.zeros((N, NODE_FEAT_DIM), dtype=torch.float32)

    data = Data(x=x_tensor, edge_index=edge_index, edge_attr=edge_attr)

    lib = state.lib_state
    placed = sorted(lib.board.items(), key=lambda kv: (kv[0][0], kv[0][1]))
    pos_to_idx: Dict[Tuple[int, int], int] = {
        (coord[0], coord[1]): i for i, (coord, _) in enumerate(placed)
    }
    data.node_tiles = [tile for _, tile in placed]
    data.last_placed_node_idx = raw['last_placed_node_idx']
    oc = raw['origin_coord']
    data.origin_coord = (oc[0], oc[1]) if N > 0 else (0, 0)
    data.x_gps = x_gps
    data.edge_attr_model = edge_attr[:, :EDGE_MODEL_DIM].clone()
    data.rwpe = compute_rwpe(edge_index, N, RWPE_DIM)
    data.dist_cats = compute_local_distance_categories(pos_to_idx, N)
    return data


def _build_graph_from_state_python(
    state: CarcassonneGameState,
    *,
    use_gps: bool = False,
) -> Data:
    """Pure-Python graph construction (fallback / legacy)."""
    lib    = state.lib_state
    player = lib.current_player
    enemy  = 1 - player

    placed: List[Tuple[int, int, Tile]] = sorted(
        ((r, c, t) for (r, c), t in lib.board.items()),
        key=lambda x: (x[0], x[1]),
    )

    N = len(placed)
    pos_to_idx: Dict[Tuple[int, int], int] = {
        (r, c): i for i, (r, c, _) in enumerate(placed)
    }

    tile_meeple: Dict[Tuple[int, int], Tuple[int, int, int]] = {}
    for p_idx in range(2):
        for mp in lib.placed_meeples[p_idx]:
            r = mp.coordinate_with_side.coordinate.row
            c = mp.coordinate_with_side.coordinate.column
            s = mp.coordinate_with_side.side
            tile_meeple[(r, c)] = (p_idx, mp.meeple_type, s)

    last_placed_pos: Optional[Tuple[int, int]] = None
    if lib.last_tile_action is not None:
        last_placed_pos = (
            lib.last_tile_action.coordinate.row,
            lib.last_tile_action.coordinate.column,
        )
    last_placed_node_idx: int = (
        pos_to_idx.get(last_placed_pos, -1)
        if last_placed_pos is not None else -1
    )

    node_tiles: List[Tile] = [tile for _, _, tile in placed]

    def _meeple_args(r, c):
        mi = tile_meeple.get((r, c))
        return mi if mi is not None else (-1, -1, -1)

    def _common_kwargs(r, c, tile):
        mo, mt, ms = _meeple_args(r, c)
        return dict(
            tile=tile,
            is_just_placed=((r, c) == last_placed_pos),
            meeple_owner=mo,
            meeple_type=mt,
            meeple_side=ms,
            player_meeples=lib.meeples[player],
            player_abbots=lib.abbots[player],
            player_big_meeples=lib.big_meeples[player],
            enemy_meeples=lib.meeples[enemy],
            enemy_abbots=lib.abbots[enemy],
            enemy_big_meeples=lib.big_meeples[enemy],
            is_meeples_phase=(lib.phase == GamePhase.MEEPLES),
        )

    if N > 0:
        node_feats = np.zeros((N, NODE_FEAT_DIM), dtype=np.float32)
        gps_feats  = np.zeros((N, GPS_NODE_FEAT_DIM), dtype=np.float32) if use_gps else None

        for i, (r, c, tile) in enumerate(placed):
            kw = _common_kwargs(r, c, tile)
            node_feats[i] = encode_tile_node_features(**kw)
            if use_gps:
                gps_feats[i] = encode_gps_node_features(**kw)

        x_tensor = torch.from_numpy(node_feats)
        origin_coord: Optional[Tuple[int, int]] = (placed[0][0], placed[0][1])
    else:
        x_tensor     = torch.zeros((0, NODE_FEAT_DIM), dtype=torch.float32)
        gps_feats    = np.zeros((0, GPS_NODE_FEAT_DIM), dtype=np.float32) if use_gps else None
        origin_coord = None

    edge_src:   List[int]        = []
    edge_dst:   List[int]        = []
    edge_feats: List[np.ndarray] = []

    for i, (r, c, tile) in enumerate(placed):
        for direction in CARDINAL:
            dr, dc = DIR_DELTAS[direction]
            j = pos_to_idx.get((r + dr, c + dc), -1)
            if j < 0:
                continue
            edge_src.append(i)
            edge_dst.append(j)
            edge_feats.append(get_edge_terrain(tile, direction))

    if edge_feats:
        edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
        edge_attr  = torch.from_numpy(np.stack(edge_feats, axis=0))
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr  = torch.zeros((0, EDGE_FEAT_DIM), dtype=torch.float32)

    data = Data(x=x_tensor, edge_index=edge_index, edge_attr=edge_attr)

    data.node_tiles            = node_tiles
    data.last_placed_node_idx  = last_placed_node_idx
    data.origin_coord = origin_coord if origin_coord is not None else (0, 0)

    if use_gps:
        data.x_gps = torch.from_numpy(gps_feats)
        data.edge_attr_model = edge_attr[:, :EDGE_MODEL_DIM].clone()
        data.rwpe = compute_rwpe(edge_index, N, RWPE_DIM)
        data.dist_cats = compute_local_distance_categories(pos_to_idx, N)

    return data


def batch_dist_cats(data_list: List[Data]) -> Tensor:
    """Merge per-graph dist_cats into a single batched dist_cats tensor.

    Each per-graph dist_cats has flat indices i*N_g + j, where N_g is the
    number of nodes in that graph.  This function recomputes them relative to
    the total number of batched nodes N_total, accounting for the node offset
    of each graph in the batch.
    """
    all_cats: List[Tensor] = []
    all_flat: List[Tensor] = []
    offset = 0
    n_total = sum(d.x_gps.shape[0] for d in data_list)

    for d in data_list:
        N_g = d.x_gps.shape[0]
        dc = getattr(d, 'dist_cats', None)
        if dc is not None and dc.shape[1] > 0:
            cat_ids = dc[0]          # (K,)
            flat_idx = dc[1]         # (K,) encoded as i*N_g + j
            local_i = flat_idx // N_g
            local_j = flat_idx - local_i * N_g
            global_flat = (local_i + offset) * n_total + (local_j + offset)
            all_cats.append(cat_ids)
            all_flat.append(global_flat)
        offset += N_g

    if not all_cats:
        return torch.zeros((2, 0), dtype=torch.long)

    return torch.stack([torch.cat(all_cats), torch.cat(all_flat)], dim=0)


def build_graph_from_state(
    state: CarcassonneGameState,
    *,
    use_gps: bool = False,
) -> Data:
    """
    Convert a CarcassonneGameState into a PyG Data object.

    Parameters
    ----------
    use_gps : bool
        If True, encode rotation-invariant GPS features (canonical terrain,
        RWPE, terrain-only model edges).  If False, use legacy features.

    The 7-dim edge_attr (with direction bits) is always stored for BFS.
    When use_gps=True, additional attributes are set:
        data.x_gps         — (N, GPS_NODE_FEAT_DIM) invariant node features
        data.rwpe           — (N, RWPE_DIM) random walk positional encoding
        data.edge_attr_model — (E, 3) terrain-only edges for the GPS model
    """
    if use_gps and _HAS_CPP_GRAPH_BUILD:
        return _build_graph_from_state_cpp(state)
    return _build_graph_from_state_python(state, use_gps=use_gps)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Graph traversal helpers (coordinate-free)
# ─────────────────────────────────────────────────────────────────────────────

def get_relative_pos_map(data: Data) -> Dict[Tuple[int, int], int]:
    """BFS from node 0 using direction bits in edge features → (rel_r, rel_c) → node_idx."""
    if hasattr(data, '_rel_pos_map') and data._rel_pos_map is not None:
        return data._rel_pos_map

    N = int(data.num_nodes)
    if N == 0:
        data._rel_pos_map = {}
        return data._rel_pos_map

    edge_index = data.edge_index
    edge_attr  = data.edge_attr

    adj: List[List[Tuple[int, int]]] = [[] for _ in range(N)]
    for e in range(edge_index.shape[1]):
        u        = int(edge_index[0, e].item())
        v        = int(edge_index[1, e].item())
        dir_bits = edge_attr[e, 3:7]
        if dir_bits.sum().item() > 0:
            dir_idx   = int(dir_bits.argmax().item())
            direction = _IDX_TO_DIRECTION[dir_idx]
            adj[u].append((v, direction))

    node_to_rel: Dict[int, Tuple[int, int]] = {0: (0, 0)}
    queue: deque = deque([0])
    while queue:
        u = queue.popleft()
        for v, direction in adj[u]:
            if v not in node_to_rel:
                dr, dc = DIR_DELTAS[direction]
                node_to_rel[v] = (node_to_rel[u][0] + dr, node_to_rel[u][1] + dc)
                queue.append(v)

    rel_pos_map: Dict[Tuple[int, int], int] = {
        pos: idx for idx, pos in node_to_rel.items()
    }
    data._rel_pos_map    = rel_pos_map
    data._node_to_rel    = node_to_rel
    return rel_pos_map


def _get_node_to_rel(data: Data) -> Dict[int, Tuple[int, int]]:
    if not hasattr(data, '_node_to_rel') or data._node_to_rel is None:
        get_relative_pos_map(data)
    return data._node_to_rel


def has_edge_in_direction(data: Data, node_u: int, direction: int) -> bool:
    dir_bit = 3 + _DIRECTION_IDX[direction]
    edge_index = data.edge_index
    edge_attr  = data.edge_attr
    for e in range(edge_index.shape[1]):
        if int(edge_index[0, e].item()) == node_u and edge_attr[e, dir_bit].item() > 0.5:
            return True
    return False


def get_graph_legal_tile_placements(
    data: Data,
    game_state: CarcassonneGameState,
) -> List[TileAction]:
    """Generate all legal tile placements using the graph structure alone."""
    lib_state = game_state.lib_state

    origin_coord = getattr(data, 'origin_coord', None)
    if data.num_nodes == 0 or origin_coord is None:
        tile = lib_state.next_tile
        starting = lib_state.starting_position
        return [TileAction(tile=tile.turn(0), coordinate=starting, tile_rotations=0)]

    origin_r, origin_c = origin_coord
    rel_pos_map  = get_relative_pos_map(data)
    node_to_rel  = _get_node_to_rel(data)
    node_tiles   = data.node_tiles
    tile_to_play = lib_state.next_tile
    N            = int(data.num_nodes)

    occupied_sides: Set[Tuple[int, int]] = set()
    edge_index = data.edge_index
    edge_attr  = data.edge_attr
    for e in range(edge_index.shape[1]):
        u        = int(edge_index[0, e].item())
        dir_bits = edge_attr[e, 3:7]
        if dir_bits.sum().item() > 0:
            dir_idx   = int(dir_bits.argmax().item())
            occupied_sides.add((u, _IDX_TO_DIRECTION[dir_idx]))

    slot_neighbors: Dict[Tuple[int, int], Dict[int, Tile]] = {}
    for node_u in range(N):
        rel_u = node_to_rel[node_u]
        for direction in CARDINAL:
            if (node_u, direction) not in occupied_sides:
                dr, dc      = DIR_DELTAS[direction]
                slot_coord  = (rel_u[0] + dr, rel_u[1] + dc)
                opp = OPPOSITE_DIR[direction]
                if slot_coord not in slot_neighbors:
                    slot_neighbors[slot_coord] = {}
                slot_neighbors[slot_coord][opp] = node_tiles[node_u]

    legal_actions: List[TileAction] = []
    for slot_coord, nb_by_dir in slot_neighbors.items():
        top    = nb_by_dir.get(Side.TOP)
        right  = nb_by_dir.get(Side.RIGHT)
        bottom = nb_by_dir.get(Side.BOTTOM)
        left   = nb_by_dir.get(Side.LEFT)

        for tile_turns in range(4):
            rotated = tile_to_play.turn(tile_turns)
            if TileFitter.fits(
                rotated,
                top=top, right=right, bottom=bottom, left=left,
                game_state=lib_state,
            ):
                abs_coord = Coordinate(
                    row    = origin_r + slot_coord[0],
                    column = origin_c + slot_coord[1],
                )
                legal_actions.append(
                    TileAction(tile=rotated, coordinate=abs_coord, tile_rotations=tile_turns)
                )

    return legal_actions


# ─────────────────────────────────────────────────────────────────────────────
# 7. Legacy GNN Model (GATv2Conv only)
# ─────────────────────────────────────────────────────────────────────────────

class GNNCarcassonneModel(nn.Module):
    """Legacy 4-layer GATv2Conv backbone (not rotation-invariant)."""

    def __init__(self) -> None:
        super().__init__()

        self.convs: nn.ModuleList = nn.ModuleList()
        self.norms: nn.ModuleList = nn.ModuleList()
        in_ch = NODE_FEAT_DIM
        for _ in range(4):
            self.convs.append(
                GATv2Conv(
                    in_channels=in_ch,
                    out_channels=GNN_HIDDEN,
                    heads=1,
                    concat=False,
                    edge_dim=EDGE_FEAT_DIM,
                    add_self_loops=True,
                    fill_value=0.0,
                )
            )
            self.norms.append(nn.LayerNorm(GNN_HIDDEN))
            in_ch = GNN_HIDDEN

        self.value_mlp = nn.Sequential(
            nn.Linear(GNN_HIDDEN, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 1),
        )

        self.tile_scorer = nn.Sequential(
            nn.Linear(TILE_ACTION_FEAT_DIM, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 1),
        )

        self.meeple_scorer = nn.Sequential(
            nn.Linear(MEEPLE_ACTION_FEAT_DIM, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 1),
        )

        self.pass_bias = nn.Parameter(torch.zeros(1))

        for scorer in (self.tile_scorer, self.meeple_scorer):
            final = scorer[-1]
            if isinstance(final, nn.Linear):
                nn.init.zeros_(final.weight)
                nn.init.zeros_(final.bias)

    def encode(self, data: Data) -> Tensor:
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        for conv, norm in zip(self.convs, self.norms):
            if x.shape[0] == 0:
                x = torch.zeros((0, GNN_HIDDEN), dtype=torch.float32, device=x.device)
                continue
            x = conv(x, edge_index, edge_attr)
            x = norm(x)
            x = F.leaky_relu(x)
        return x

    def predict_value(self, node_embs: Tensor, data: Data) -> Tensor:
        device = node_embs.device
        if hasattr(data, 'batch') and data.batch is not None:
            batch_vec = data.batch
            num_graphs = int(data.num_graphs)
        else:
            batch_vec = torch.zeros(node_embs.shape[0], dtype=torch.long, device=device)
            num_graphs = 1
        pooled = global_mean_pool(node_embs, batch_vec, size=num_graphs)
        return torch.tanh(self.value_mlp(pooled))

    def score_actions(self, node_embs, data, last_placed_idx, legal_actions):
        scores: List[Tensor] = []
        for action in legal_actions:
            if isinstance(action, TileAction):
                s = self._score_tile_action(action, node_embs, data)
            elif isinstance(action, MeepleAction):
                s = self.pass_bias.squeeze() if action.remove else \
                    self._score_meeple_action(action, node_embs, last_placed_idx)
            else:
                s = self.pass_bias.squeeze()
            scores.append(s)
        return torch.stack(scores)

    def _score_tile_action(self, action, node_embs, data):
        device = node_embs.device
        tile_feat = torch.from_numpy(encode_tile_static(action.tile)).to(device)

        neighbour_embs: List[Tensor]    = []
        neighbour_terrains: List[np.ndarray] = []
        origin_coord = getattr(data, 'origin_coord', None)
        if node_embs.shape[0] > 0 and origin_coord is not None:
            origin_r, origin_c = origin_coord
            rel_pos_map = get_relative_pos_map(data)
            action_rel_r = action.coordinate.row    - origin_r
            action_rel_c = action.coordinate.column - origin_c
            for direction in CARDINAL:
                dr, dc = DIR_DELTAS[direction]
                nb_node = rel_pos_map.get((action_rel_r + dr, action_rel_c + dc))
                if nb_node is not None:
                    neighbour_embs.append(node_embs[nb_node])
                    neighbour_terrains.append(get_edge_terrain(action.tile, direction))

        if neighbour_embs:
            nb_emb_mean = torch.stack(neighbour_embs).mean(dim=0)
            nb_terrain_sum = torch.from_numpy(np.stack(neighbour_terrains).sum(axis=0)).to(device)
        else:
            nb_emb_mean    = torch.zeros(GNN_HIDDEN,    device=device)
            nb_terrain_sum = torch.zeros(EDGE_FEAT_DIM, device=device)

        feat = torch.cat([tile_feat, nb_emb_mean, nb_terrain_sum])
        return self.tile_scorer(feat.unsqueeze(0)).squeeze()

    def _score_meeple_action(self, action, node_embs, last_placed_idx):
        device = node_embs.device
        last_emb = node_embs[last_placed_idx] if last_placed_idx >= 0 and node_embs.shape[0] > 0 \
            else torch.zeros(GNN_HIDDEN, device=device)

        mtype_feat = torch.zeros(NUM_MEEPLE_TYPES, device=device)
        if 0 <= action.meeple_type < NUM_MEEPLE_TYPES:
            mtype_feat[action.meeple_type] = 1.0

        mside_feat = torch.zeros(NUM_SIDES, device=device)
        side = action.coordinate_with_side.side
        if 0 <= side < NUM_SIDES:
            mside_feat[side] = 1.0

        feat = torch.cat([last_emb, mtype_feat, mside_feat])
        return self.meeple_scorer(feat.unsqueeze(0)).squeeze()

    def forward(self, data, legal_actions):
        node_embs = self.encode(data)
        value = self.predict_value(node_embs, data).squeeze().item()
        scores = self.score_actions(node_embs, data, data.last_placed_node_idx, legal_actions)
        probs  = torch.softmax(scores, dim=0)
        policy = normalize_dict({a: p.item() for a, p in zip(legal_actions, probs)})
        if policy is None:
            policy = {a: 1.0 / len(legal_actions) for a in legal_actions}
        return {'policy': policy, 'value': value, 'scores': scores}


# ─────────────────────────────────────────────────────────────────────────────
# 8. GPS Graph Transformer Model
# ─────────────────────────────────────────────────────────────────────────────

class GPSLayer(nn.Module):
    """
    Single GPS (General, Powerful, Scalable) layer.
    Combines local message passing (GATv2Conv) with global multi-head
    self-attention and a feed-forward network, all with residual connections
    and layer normalisation.
    """

    def __init__(self, hidden: int, heads: int, edge_dim: int) -> None:
        super().__init__()
        self.num_heads = heads

        # Local message passing
        self.local_conv = GATv2Conv(
            in_channels=hidden,
            out_channels=hidden,
            heads=heads,
            concat=False,
            edge_dim=edge_dim,
            add_self_loops=True,
            fill_value=0.0,
        )
        self.norm_local = nn.LayerNorm(hidden)

        # Global self-attention
        self.global_attn = nn.MultiheadAttention(
            embed_dim=hidden,
            num_heads=heads,
            batch_first=True,
        )
        self.norm_global = nn.LayerNorm(hidden)

        # Learned per-head bias for each local distance category
        self.dist_bias = nn.Embedding(NUM_DIST_CATEGORIES, heads)

        # Feed-forward
        self.ffn = nn.Sequential(
            nn.Linear(hidden, hidden * 4),
            nn.GELU(),
            nn.Linear(hidden * 4, hidden),
        )
        self.norm_ffn = nn.LayerNorm(hidden)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        batch: Tensor,
        num_graphs: int,
        dist_cats: Optional[Tensor] = None,
    ) -> Tensor:
        N = x.shape[0]
        if N == 0:
            return x

        # ── local message passing + residual ──────────────────────────────
        h_local = self.local_conv(x, edge_index, edge_attr)
        x = self.norm_local(x + h_local)

        # ── global self-attention (per graph) + residual ──────────────────
        counts = torch.zeros(num_graphs, dtype=torch.long, device=x.device)
        counts.scatter_add_(0, batch, torch.ones(N, dtype=torch.long, device=x.device))
        max_n = int(counts.max().item())

        nonempty_mask = counts > 0
        nonempty_idx = nonempty_mask.nonzero(as_tuple=True)[0]
        n_nonempty = nonempty_idx.shape[0]

        remap = torch.zeros(num_graphs, dtype=torch.long, device=x.device)
        remap[nonempty_idx] = torch.arange(n_nonempty, dtype=torch.long, device=x.device)
        batch_compact = remap[batch]
        counts_compact = counts[nonempty_idx]

        graph_starts = torch.zeros(n_nonempty, dtype=torch.long, device=x.device)
        graph_starts[1:] = counts_compact[:-1].cumsum(0)
        within_pos = torch.arange(N, dtype=torch.long, device=x.device) - graph_starts[batch_compact]

        padded = torch.zeros(n_nonempty, max_n, x.shape[1], dtype=x.dtype, device=x.device)
        key_padding_mask = torch.ones(n_nonempty, max_n, dtype=torch.bool, device=x.device)

        padded[batch_compact, within_pos] = x
        key_padding_mask[batch_compact, within_pos] = False

        # ── build distance-aware attention bias ───────────────────────────
        # Convert bool key_padding_mask to a float attn_mask so both masks
        # share the same dtype (avoids PyTorch deprecation warning).
        H = self.num_heads
        pad_bias = key_padding_mask.unsqueeze(1).unsqueeze(2).expand(
            -1, H, max_n, -1
        )  # (B, H, max_n, max_n) — broadcast over query dim
        # True (padded) → -inf, False (real) → 0.0
        combined_mask = torch.zeros(
            n_nonempty, H, max_n, max_n, dtype=x.dtype, device=x.device
        )
        combined_mask.masked_fill_(pad_bias, float('-inf'))

        if dist_cats is not None and dist_cats.shape[1] > 0:
            cat_ids = dist_cats[0]       # (K,) category indices
            flat_idx = dist_cats[1]      # (K,) flat i*N_total + j indices

            global_i = flat_idx // N     # global node index of source
            global_j = flat_idx - global_i * N  # global node index of target

            bi = batch_compact[global_i]
            li = within_pos[global_i]
            lj = within_pos[global_j]

            bias_vals = self.dist_bias(cat_ids)  # (K, num_heads)

            bi_exp = bi.unsqueeze(1).expand(-1, H)   # (K, H)
            h_exp = torch.arange(H, device=x.device).unsqueeze(0).expand(len(bi), -1)
            li_exp = li.unsqueeze(1).expand(-1, H)
            lj_exp = lj.unsqueeze(1).expand(-1, H)

            combined_mask[bi_exp.reshape(-1), h_exp.reshape(-1),
                          li_exp.reshape(-1), lj_exp.reshape(-1)] += bias_vals.reshape(-1)

        attn_mask = combined_mask.reshape(n_nonempty * H, max_n, max_n)

        attn_out, _ = self.global_attn(
            padded, padded, padded,
            attn_mask=attn_mask,
        )

        h_global = attn_out[batch_compact, within_pos]
        x = self.norm_global(x + h_global)

        # ── FFN + residual ────────────────────────────────────────────────
        x = self.norm_ffn(x + self.ffn(x))

        return x


class MultiHeadReadout(nn.Module):
    """Learned multi-query attentive graph readout.

    K query vectors each attend over all nodes (per graph) via dot-product
    attention, producing K weighted sums that are concatenated.
    Output shape: (B, K * hidden).
    """

    def __init__(self, hidden: int, num_queries: int = 4) -> None:
        super().__init__()
        self.num_queries = num_queries
        self.queries = nn.Parameter(torch.randn(num_queries, hidden))
        self.key_proj = nn.Linear(hidden, hidden)

    def forward(self, node_embs: Tensor, batch_vec: Tensor,
                num_graphs: int) -> Tensor:
        N = node_embs.shape[0]
        device = node_embs.device
        K = self.num_queries
        H = node_embs.shape[1]

        if N == 0:
            return torch.zeros(num_graphs, K * H, dtype=node_embs.dtype, device=device)

        keys = self.key_proj(node_embs)                     # (N, H)
        attn_logits = keys @ self.queries.T                  # (N, K)

        pooled_parts: List[Tensor] = []
        for q in range(K):
            weights = pyg_softmax(attn_logits[:, q], batch_vec, num_nodes=num_graphs)  # (N,)
            weighted = node_embs * weights.unsqueeze(1)      # (N, H)
            pooled_q = torch.zeros(num_graphs, H, dtype=node_embs.dtype, device=device)
            pooled_q.scatter_add_(0, batch_vec.unsqueeze(1).expand_as(weighted), weighted)
            pooled_parts.append(pooled_q)

        return torch.cat(pooled_parts, dim=1)                # (B, K * H)


class GPSCarcassonneModel(nn.Module):
    """
    GPS Graph Transformer for Carcassonne.

    Rotation-invariant, translation-invariant, chirality-preserving.

    Architecture:
      - Input projection:  Linear(GPS_NODE_FEAT_DIM -> hidden)
      - RWPE encoder:      Linear(RWPE_DIM -> hidden), added to node features
      - Edge encoder:      Linear(EDGE_MODEL_DIM -> hidden)
      - num_layers x GPSLayer (local GATv2Conv + global MHSA + FFN)
      - Value head:        MultiHeadReadout (K=4 learned queries) -> MLP -> tanh
      - Policy head:       direction-aware tile scorer + batched action scoring
    """

    def __init__(
        self,
        gps_layers: int = GPS_LAYERS,
        gnn_hidden: int = GNN_HIDDEN,
        gps_heads: int = GPS_HEADS,
    ) -> None:
        super().__init__()
        self.gnn_hidden = gnn_hidden

        self.node_proj = nn.Linear(GPS_NODE_FEAT_DIM, gnn_hidden)
        self.rwpe_enc  = nn.Sequential(
            nn.Linear(RWPE_DIM, gnn_hidden),
            nn.GELU(),
            nn.Linear(gnn_hidden, gnn_hidden),
        )
        self.edge_enc  = nn.Linear(EDGE_MODEL_DIM, gnn_hidden)

        self.layers = nn.ModuleList([
            GPSLayer(gnn_hidden, gps_heads, edge_dim=gnn_hidden)
            for _ in range(gps_layers)
        ])

        self.value_readout = MultiHeadReadout(gnn_hidden, num_queries=4)
        self.value_mlp = nn.Sequential(
            nn.Linear(gnn_hidden * 4, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 1),
        )

        self.neighbor_proj = nn.Sequential(
            nn.Linear(gnn_hidden + EDGE_MODEL_DIM, gnn_hidden),
            nn.GELU(),
        )
        tile_feat_dim = GPS_TILE_STATIC_DIM + gnn_hidden
        self.tile_scorer = nn.Sequential(
            nn.Linear(tile_feat_dim, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 1),
        )

        meeple_feat_dim = gnn_hidden + NUM_MEEPLE_TYPES + NUM_SIDES
        self.meeple_scorer = nn.Sequential(
            nn.Linear(meeple_feat_dim, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 1),
        )

        self.pass_bias = nn.Parameter(torch.zeros(1))

        self._init_scorer_weights()

    def _init_scorer_weights(self) -> None:
        """Zero-init the final layer of each scorer so that at initialization,
        all action scores are ~0, matching pass_bias=0.  This prevents the
        untrained model from systematically preferring pass over meeple
        placement (or vice-versa)."""
        for scorer in (self.tile_scorer, self.meeple_scorer):
            final_linear = scorer[-1]
            if isinstance(final_linear, nn.Linear):
                nn.init.zeros_(final_linear.weight)
                nn.init.zeros_(final_linear.bias)

    # ── backbone ──────────────────────────────────────────────────────────

    def encode_tensors(
        self,
        x_gps: Tensor,
        rwpe: Tensor,
        edge_index: Tensor,
        edge_attr_model: Tensor,
        batch_vec: Tensor,
        dist_cats: Optional[Tensor] = None,
    ) -> Tensor:
        """Encode: accepts raw tensors, no Data objects."""
        num_graphs = int(batch_vec.max().item()) + 1
        x = self.node_proj(x_gps) + self.rwpe_enc(rwpe)
        edge_h = self.edge_enc(edge_attr_model)
        for layer in self.layers:
            x = layer(x, edge_index, edge_h, batch_vec, num_graphs, dist_cats)
        return x

    def encode(self, data: Data) -> Tensor:
        x_gps = data.x_gps
        rwpe   = data.rwpe
        edge_index = data.edge_index
        edge_attr_model = data.edge_attr_model

        N = x_gps.shape[0]
        if N == 0:
            return torch.zeros((0, self.gnn_hidden), dtype=torch.float32, device=x_gps.device)

        if hasattr(data, 'batch') and data.batch is not None:
            batch_vec = data.batch
        else:
            batch_vec = torch.zeros(N, dtype=torch.long, device=x_gps.device)

        dist_cats = getattr(data, 'dist_cats', None)
        return self.encode_tensors(x_gps, rwpe, edge_index, edge_attr_model, batch_vec, dist_cats)

    # ── value head ────────────────────────────────────────────────────────

    def predict_value(self, node_embs: Tensor, data: Data) -> Tensor:
        device = node_embs.device
        if hasattr(data, 'batch') and data.batch is not None:
            batch_vec  = data.batch
            num_graphs = int(data.num_graphs)
        else:
            batch_vec  = torch.zeros(node_embs.shape[0], dtype=torch.long, device=device)
            num_graphs = 1
        pooled = self.value_readout(node_embs, batch_vec, num_graphs)
        return torch.tanh(self.value_mlp(pooled))

    # ── action scoring ────────────────────────────────────────────────────

    def score_actions(self, node_embs, data, last_placed_idx, legal_actions):
        device = node_embs.device
        H = self.gnn_hidden

        tile_actions:   List[Tuple[int, TileAction]]   = []
        meeple_actions: List[Tuple[int, MeepleAction]] = []
        pass_indices:   List[int]                       = []

        for idx, action in enumerate(legal_actions):
            if isinstance(action, TileAction):
                tile_actions.append((idx, action))
            elif isinstance(action, MeepleAction) and not action.remove:
                meeple_actions.append((idx, action))
            else:
                pass_indices.append(idx)

        scores = torch.zeros(len(legal_actions), device=device)

        if tile_actions:
            tile_feats = self._batch_tile_features(
                [a for _, a in tile_actions], node_embs, data)
            tile_scores = self.tile_scorer(tile_feats).squeeze(-1)
            tile_idx = torch.tensor([i for i, _ in tile_actions], dtype=torch.long, device=device)
            scores.scatter_(0, tile_idx, tile_scores)

        if meeple_actions:
            meeple_feats = self._batch_meeple_features(
                [a for _, a in meeple_actions], node_embs, last_placed_idx)
            meeple_scores = self.meeple_scorer(meeple_feats).squeeze(-1)
            meeple_idx = torch.tensor([i for i, _ in meeple_actions], dtype=torch.long, device=device)
            scores.scatter_(0, meeple_idx, meeple_scores)

        if pass_indices:
            pass_idx = torch.tensor(pass_indices, dtype=torch.long, device=device)
            scores.scatter_(0, pass_idx, self.pass_bias.squeeze().expand(len(pass_indices)))

        return scores

    def _batch_tile_features(self, tile_actions: List[TileAction],
                             node_embs: Tensor, data) -> Tensor:
        """Build batched feature tensor for all tile actions at once.

        Uses direction-aware neighbor projection: for each cardinal direction
        with a neighbor, [neighbor_emb, terrain_onehot] is projected through
        neighbor_proj, then all projected neighbors are summed (permutation-
        invariant).  Feature: [tile_static(74), projected_neighbor_sum(H)].
        """
        device = node_embs.device
        H = self.gnn_hidden
        K = len(tile_actions)

        tile_static_list: List[np.ndarray] = []
        nb_input_list: List[Tensor] = []

        origin_coord = getattr(data, 'origin_coord', None)
        rel_pos_map = get_relative_pos_map(data) if (
            node_embs.shape[0] > 0 and origin_coord is not None
        ) else {}
        origin_r = origin_coord[0] if origin_coord is not None else 0
        origin_c = origin_coord[1] if origin_coord is not None else 0

        for action in tile_actions:
            tile = action.tile
            tile_static_list.append(encode_gps_tile_static(tile))

            per_nb: List[Tensor] = []
            if rel_pos_map:
                ar = action.coordinate.row    - origin_r
                ac = action.coordinate.column - origin_c
                for direction in CARDINAL:
                    dr, dc = DIR_DELTAS[direction]
                    nb_node = rel_pos_map.get((ar + dr, ac + dc))
                    if nb_node is not None:
                        terrain = tile.get_type(direction)
                        tv = torch.zeros(EDGE_MODEL_DIM, device=device)
                        tidx = _EDGE_TERRAIN_IDX.get(terrain, -1)
                        if tidx >= 0:
                            tv[tidx] = 1.0
                        per_nb.append(torch.cat([node_embs[nb_node], tv]))

            if per_nb:
                nb_input_list.append(torch.stack(per_nb))
            else:
                nb_input_list.append(torch.zeros(1, H + EDGE_MODEL_DIM, device=device))

        tile_static_t = torch.from_numpy(np.stack(tile_static_list)).to(device)  # (K, 74)

        nb_counts = [nb.shape[0] for nb in nb_input_list]
        all_nb = torch.cat(nb_input_list, dim=0)                  # (total_nb, H+3)
        all_nb_proj = self.neighbor_proj(all_nb)                   # (total_nb, H)

        nb_sum = torch.zeros(K, H, device=device)
        batch_idx = torch.cat([
            torch.full((c,), i, dtype=torch.long, device=device)
            for i, c in enumerate(nb_counts)
        ])
        nb_sum.scatter_add_(0, batch_idx.unsqueeze(1).expand_as(all_nb_proj), all_nb_proj)

        feats = torch.cat([tile_static_t, nb_sum], dim=1)         # (K, 74+H)
        return feats

    def _batch_meeple_features(self, meeple_actions: List[MeepleAction],
                               node_embs: Tensor,
                               last_placed_idx) -> Tensor:
        """Build batched feature tensor for all meeple actions at once."""
        device = node_embs.device
        H = self.gnn_hidden
        K = len(meeple_actions)

        last_emb = node_embs[last_placed_idx] if (
            last_placed_idx >= 0 and node_embs.shape[0] > 0
        ) else torch.zeros(H, device=device)

        feats = torch.zeros(K, H + NUM_MEEPLE_TYPES + NUM_SIDES, device=device)
        feats[:, :H] = last_emb.unsqueeze(0)

        for i, action in enumerate(meeple_actions):
            if 0 <= action.meeple_type < NUM_MEEPLE_TYPES:
                feats[i, H + action.meeple_type] = 1.0
            side = action.coordinate_with_side.side
            if 0 <= side < NUM_SIDES:
                feats[i, H + NUM_MEEPLE_TYPES + side] = 1.0

        return feats

    # ── full forward ──────────────────────────────────────────────────────

    def forward(self, data, legal_actions):
        node_embs = self.encode(data)
        value = self.predict_value(node_embs, data).squeeze().item()
        scores = self.score_actions(node_embs, data, data.last_placed_node_idx, legal_actions)
        probs  = torch.softmax(scores, dim=0)
        policy = normalize_dict({a: p.item() for a, p in zip(legal_actions, probs)})
        if policy is None:
            policy = {a: 1.0 / len(legal_actions) for a in legal_actions}
        return {'policy': policy, 'value': value, 'scores': scores}


# ─────────────────────────────────────────────────────────────────────────────
# 9. Agent
# ─────────────────────────────────────────────────────────────────────────────

class CarcassonneGNNAgent(MCTSMLAgent[CarcassonneGameState]):
    """
    Carcassonne agent backed by either the legacy GNN or GPS Graph Transformer.

    Parameters
    ----------
    model_type : str
        'gps' (default) → GPSCarcassonneModel (rotation-invariant)
        'legacy'         → GNNCarcassonneModel (original GATv2Conv)
    """

    def __init__(
        self,
        name: str,
        summary_writer,
        config: Optional[MCTSMLAgentConfig] = None,
        model_type: str = 'gps',
        model_size: str = 'default',
    ) -> None:
        self._model_type = model_type
        if model_type == 'gps':
            preset = MODEL_SIZE_PRESETS.get(model_size, MODEL_SIZE_PRESETS['default'])
            model = GPSCarcassonneModel(**preset)
        else:
            model = GNNCarcassonneModel()
        super().__init__(name, model, summary_writer, config)
        self._traced_encode = None
        if model_type == 'gps':
            self._try_trace_encode()

    def _try_trace_encode(self):
        """Attempt to compile/optimize the GPS encoder for faster inference.

        torch.compile is disabled because encode_tensors uses .item() to
        extract num_graphs / max_n as Python ints (required for dynamic
        tensor allocation in the attention padding).  This causes graph
        breaks that split the computation into many tiny subgraphs, adding
        recompilation overhead on every new batch geometry — net-negative
        compared to eager mode.

        The encode_tensors refactoring (raw tensor args, no Data object
        attribute lookups, unified GPSLayer attention path) already provides
        the meaningful speedup.
        """
        self._traced_encode = None

    @property
    def use_gps(self) -> bool:
        return self._model_type == 'gps'

    def _state_to_model_input(self, state: CarcassonneGameState):
        return build_graph_from_state(state, use_gps=self.use_gps)

    def _pred_policy_to_action_probs(self, pred_policy_logit, game_legal_actions):
        raise NotImplementedError

    def _action_probs_to_label(self, action_probs):
        raise NotImplementedError

    def batched_evaluate(
        self,
        states_and_actions: List[Tuple[Any, List[Any]]],
    ) -> List[Dict[str, Any]]:
        """Evaluate a batch of (state, legal_actions) pairs.

        This is the core NN inference function shared by both per-game MCTS
        and the cross-game batching coordinator.  Supports FP16 on CUDA via
        ``torch.autocast``.
        """
        if not states_and_actions:
            return []

        data_list = [
            build_graph_from_state(state, use_gps=self.use_gps)
            for state, _ in states_and_actions
        ]

        _exclude = ['node_tiles', 'origin_coord', 'last_placed_node_idx', 'dist_cats']
        batched = Batch.from_data_list(
            data_list, exclude_keys=_exclude
        ).to(self.device)
        batched.dist_cats = batch_dist_cats(data_list).to(self.device)

        self.model.eval()
        use_amp = (self.device.type == 'cuda')
        with torch.no_grad(), torch.autocast(
            device_type=self.device.type, dtype=torch.float16, enabled=use_amp
        ):
            encode_fn = self._traced_encode or self.model.encode_tensors
            N = batched.x_gps.shape[0]
            if N == 0:
                node_embs = torch.zeros(
                    (0, self.model.gnn_hidden), dtype=torch.float32,
                    device=self.device)
            else:
                batch_vec = batched.batch if batched.batch is not None \
                    else torch.zeros(N, dtype=torch.long, device=self.device)
                dist_cats = getattr(batched, 'dist_cats', None)
                node_embs = encode_fn(
                    batched.x_gps, batched.rwpe, batched.edge_index,
                    batched.edge_attr_model, batch_vec, dist_cats,
                )
            all_values = self.model.predict_value(node_embs, batched)

        node_embs = node_embs.float()
        all_values = all_values.float()

        results: List[Dict[str, Any]] = []
        for i, ((_state, legal_actions), data_i) in enumerate(
            zip(states_and_actions, data_list)
        ):
            value = all_values[i].item()

            mask = batched.batch == i
            graph_embs = node_embs[mask]

            with torch.no_grad():
                scores = self.model.score_actions(
                    graph_embs, data_i,
                    data_i.last_placed_node_idx, legal_actions,
                )
                probs = torch.softmax(scores, dim=0)

            policy = normalize_dict(
                {a: p.item() for a, p in zip(legal_actions, probs)}
            )
            if policy is None:
                policy = {a: 1.0 / len(legal_actions) for a in legal_actions}

            results.append({'policy': policy, 'value': value})

        return results

    def _prepare_mcts(self) -> MCTS:
        def inference_fn(state: CarcassonneGameState, all_legal_actions: List[Any]):
            data = build_graph_from_state(state, use_gps=self.use_gps).to(self.device)

            self.model.eval()
            with torch.no_grad():
                with self.model_summarizer:
                    result = self.model(data, all_legal_actions)

            return {'policy': result['policy'], 'value': result['value']}

        def batched_inference_fn(
            states_and_actions: List[Tuple[Any, List[Any]]],
        ) -> List[Dict[str, Any]]:
            return self.batched_evaluate(states_and_actions)

        return MCTS(inference_fn, self.config.mcts_config,
                     model_batch=batched_inference_fn)

    # ── parameter groups ────────────────────────────────────────────────────

    _PARAM_GROUPS = [
        ("backbone",     ["node_proj", "rwpe_enc", "edge_enc", "layers"]),
        ("value_head",   ["value_readout", "value_mlp"]),
        ("policy_head",  ["neighbor_proj", "tile_scorer", "meeple_scorer", "pass_bias"]),
    ]

    def _build_optimizer(self):
        grouped: Dict[str, List[nn.Parameter]] = {g: [] for g, _ in self._PARAM_GROUPS}
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            assigned = False
            for group_name, prefixes in self._PARAM_GROUPS:
                if any(name.startswith(pfx) or name.startswith(pfx + ".") for pfx in prefixes):
                    grouped[group_name].append(param)
                    assigned = True
                    break
            if not assigned:
                grouped["backbone"].append(param)

        param_groups = [
            {"params": grouped[g], "group_name": g}
            for g, _ in self._PARAM_GROUPS if grouped[g]
        ]
        self.optimizer = torch.optim.SGD(
            param_groups,
            lr=self.config.learning_rate,
            momentum=self.config.momentum,
            weight_decay=self.config.weight_decay,
        )

    def _classify_param(self, name: str) -> str:
        for group_name, prefixes in self._PARAM_GROUPS:
            if any(name.startswith(pfx) or name.startswith(pfx + ".") for pfx in prefixes):
                return group_name
        return "backbone"

    # ── diagnostics ──────────────────────────────────────────────────────

    def _log_param_grad_stats(self, current_iteration: int, epoch: int):
        """Print per-layer mean/std of parameters and their gradients."""
        header = f"[Iter {current_iteration} | Epoch {epoch}] Parameter & Gradient Statistics"
        lines = [header, "-" * len(header)]
        lines.append(f"{'Layer':<45} {'Param μ':>10} {'Param σ':>10} {'Grad μ':>10} {'Grad σ':>10} {'|G|/|P|':>10}")
        lines.append("-" * 97)

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            p = param.data
            p_mean = p.mean().item()
            p_std = p.std().item() if p.numel() > 1 else 0.0
            if param.grad is not None:
                g = param.grad.data
                g_mean = g.mean().item()
                g_std = g.std().item() if g.numel() > 1 else 0.0
                ratio = g.norm().item() / (p.norm().item() + 1e-12)
            else:
                g_mean = g_std = ratio = 0.0
            lines.append(
                f"{name:<45} {p_mean:>10.4e} {p_std:>10.4e} "
                f"{g_mean:>10.4e} {g_std:>10.4e} {ratio:>10.4e}"
            )
        print("\n".join(lines), flush=True)

    def _log_convergence_diagnostic(self, current_iteration: int, epoch: int):
        """Print per-group convergence diagnostic.

        For each parameter group (backbone, value_head, policy_head), reports:
          |ΔW|/|W|  — effective update magnitude (how much weights changed
                      relative to their current scale)
          |G|/|W|   — gradient-to-weight ratio (raw gradient pressure before
                      Adam scaling)

        A group whose |ΔW|/|W| is an order of magnitude larger than another is
        converging faster (or diverging).  Healthy values are ~1e-3 to 1e-2.
        If one head has |ΔW|/|W| >> backbone, the backbone may be underfitting.
        """
        lr = self.optimizer.param_groups[0]['lr']
        groups: Dict[str, Dict[str, float]] = {}

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            g_name = self._classify_param(name)
            if g_name not in groups:
                groups[g_name] = {"w_sq": 0.0, "g_sq": 0.0, "update_sq": 0.0, "n_params": 0}
            acc = groups[g_name]
            w = param.data
            acc["w_sq"] += w.pow(2).sum().item()
            acc["n_params"] += w.numel()
            if param.grad is not None:
                g = param.grad.data
                acc["g_sq"] += g.pow(2).sum().item()
                sgd_step = lr * g
                acc["update_sq"] += sgd_step.pow(2).sum().item()

        header = f"[Iter {current_iteration} | Epoch {epoch}] Convergence Diagnostic (lr={lr:.2e})"
        lines = [header, "-" * len(header)]
        lines.append(f"{'Group':<15} {'#Params':>10} {'|W|':>12} {'|G|':>12} "
                      f"{'|ΔW|':>12} {'|G|/|W|':>10} {'|ΔW|/|W|':>10}")
        lines.append("-" * 93)

        for g_name, _ in self._PARAM_GROUPS:
            if g_name not in groups:
                continue
            acc = groups[g_name]
            w_norm = math.sqrt(acc["w_sq"])
            g_norm = math.sqrt(acc["g_sq"])
            u_norm = math.sqrt(acc["update_sq"])
            gw = g_norm / (w_norm + 1e-12)
            uw = u_norm / (w_norm + 1e-12)
            lines.append(
                f"{g_name:<15} {acc['n_params']:>10d} {w_norm:>12.4e} {g_norm:>12.4e} "
                f"{u_norm:>12.4e} {gw:>10.4e} {uw:>10.4e}"
            )
            step = current_iteration * 2 + (1 if epoch > 0 else 0)
            self.summary_writer.add_scalar(f'{self.name}_{g_name}_grad_to_weight', gw, step)
            self.summary_writer.add_scalar(f'{self.name}_{g_name}_update_to_weight', uw, step)

        print("\n".join(lines), flush=True)

    # ── training loop ────────────────────────────────────────────────────

    def train(
        self,
        train_data: List[Tuple[CarcassonneGameState, Dict[Any, float], float]],
        current_iteration: int,
    ) -> None:
        self._build_optimizer()
        self._apply_lr_for_iteration(current_iteration)

        n_steps = self.config.train_steps_per_iter
        bs = self.config.train_batch_size
        buf = list(train_data)

        total_loss   = 0.0
        total_ploss  = 0.0
        total_vloss  = 0.0

        self._log_param_grad_stats(current_iteration, 0)
        self._log_convergence_diagnostic(current_iteration, 0)

        for step in tqdm(range(n_steps), desc='Training', mininterval=1):
            batch = random.choices(buf, k=bs)
            loss_val, p_loss, v_loss = self._train_step_gnn(batch)

            total_loss  += loss_val
            total_ploss += p_loss
            total_vloss += v_loss

        if n_steps > 0:
            global_step = current_iteration
            avg_loss  = total_loss  / n_steps
            avg_ploss = total_ploss / n_steps
            avg_vloss = total_vloss / n_steps
            lr = self.optimizer.param_groups[0]['lr']
            self.summary_writer.add_scalar(f'{self.name}_policy_loss', avg_ploss, global_step)
            self.summary_writer.add_scalar(f'{self.name}_value_loss',  avg_vloss, global_step)
            self.summary_writer.add_scalar(f'{self.name}_loss',        avg_loss,  global_step)
            self.summary_writer.add_scalar(f'{self.name}_lr',          lr,        global_step)
            print(f"  [Train] loss={avg_loss:.4f} (policy={avg_ploss:.4f}, value={avg_vloss:.4f}), "
                  f"lr={lr:.2e}, steps={n_steps}, buffer={len(buf)}", flush=True)

        self._log_param_grad_stats(current_iteration, n_steps)
        self._log_convergence_diagnostic(current_iteration, n_steps)

    def _train_step_gnn(
        self,
        batch: List[Tuple[CarcassonneGameState, Dict[Any, float], float]],
    ) -> Tuple[float, float, float]:
        self.model.train()
        self.optimizer.zero_grad()

        raw_data_list = [build_graph_from_state(state, use_gps=self.use_gps) for state, _, _ in batch]

        keep = [i for i, d in enumerate(raw_data_list) if d.x_gps.shape[0] > 0]
        if not keep:
            return 0.0, 0.0, 0.0
        data_list = [raw_data_list[i] for i in keep]
        batch_filtered = [batch[i] for i in keep]

        _exclude = ['node_tiles', 'origin_coord', 'last_placed_node_idx', 'dist_cats']
        batched   = Batch.from_data_list(data_list, exclude_keys=_exclude).to(self.device)
        batched.dist_cats = batch_dist_cats(data_list).to(self.device)

        node_embs = self.model.encode(batched)
        values    = self.model.predict_value(node_embs, batched)

        all_scores_parts:  List[Tensor] = []
        all_targets:       List[float]  = []
        action_batch_idx:  List[int]    = []

        for i, (state, action_probs, _) in enumerate(batch_filtered):
            sample_mask      = (batched.batch == i)
            sample_node_embs = node_embs[sample_mask]
            data_i           = data_list[i]

            legal_actions = list(action_probs.keys())
            if not legal_actions:
                continue

            scores_i = self.model.score_actions(
                sample_node_embs,
                data_i,
                data_i.last_placed_node_idx,
                legal_actions,
            )

            all_scores_parts.append(scores_i)
            all_targets.extend(action_probs[a] for a in legal_actions)
            action_batch_idx.extend([i] * len(legal_actions))

        if not all_scores_parts:
            return 0.0, 0.0, 0.0

        B = len(batch_filtered)
        all_scores  = torch.cat(all_scores_parts)
        target_tens = torch.tensor(all_targets, dtype=torch.float32, device=self.device)
        bidx = torch.tensor(action_batch_idx, dtype=torch.long, device=self.device)

        probs       = pyg_softmax(all_scores, bidx, num_nodes=B)
        policy_loss = -(target_tens * torch.log(probs + 1e-8)).sum() / B

        value_targets = torch.tensor(
            [v for _, _, v in batch_filtered], dtype=torch.float32, device=self.device
        ).view(-1, 1)
        value_loss = F.mse_loss(values, value_targets)

        loss = policy_loss + value_loss
        loss.backward()

        if self.config.gradient_clipping > 0:
            nn.utils.clip_grad_value_(self.model.parameters(), self.config.gradient_clipping)

        self.optimizer.step()

        return loss.item(), policy_loss.item(), value_loss.item()

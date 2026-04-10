# Carcassonne AlphaZero — Complete Technical Reference

This document is a self-contained reference for the entire Carcassonne AlphaZero
system: the game engine, the graph-based state representation, the GPS Graph
Transformer model, the training pipeline, and the MCTS search.

All neural network code lives in
[`agents/carcassonne_gnn_agent.py`](agents/carcassonne_gnn_agent.py).
The game engine lives under [`carcassonne/`](carcassonne/).
The training loop is [`carcassonne_train.py`](carcassonne_train.py).

---

## Table of Contents

**Part I — Game Engine**

1. [Game Overview](#1-game-overview)
2. [Tile Structure](#2-tile-structure)
3. [Board Representation](#3-board-representation)
4. [Game Phases and Turn Flow](#4-game-phases-and-turn-flow)
5. [Legal Move Generation](#5-legal-move-generation)
6. [Scoring Rules](#6-scoring-rules)
7. [State Transitions](#7-state-transitions)

**Part II — Neural Architecture**

8. [Why a Graph Transformer?](#8-why-a-graph-transformer)
9. [Tile Vocabulary and Node/Edge Creation](#9-tile-vocabulary-and-nodeedge-creation)
10. [Invariant Feature Encoding (GPS)](#10-invariant-feature-encoding-gps)
11. [Random Walk Positional Encoding (RWPE)](#11-random-walk-positional-encoding-rwpe)
12. [Edge Feature Split](#12-edge-feature-split)
13. [GPS Graph Transformer Architecture](#13-gps-graph-transformer-architecture)
14. [Value Head](#14-value-head)
15. [Policy Head — Action Scorer](#15-policy-head--action-scorer)

**Part III — Training and Search**

16. [AlphaZero Training Pipeline](#16-alphazero-training-pipeline)
17. [Monte Carlo Tree Search (MCTS)](#17-monte-carlo-tree-search-mcts)
18. [Variable-K Training Loop](#18-variable-k-training-loop)

**Part IV — Properties and Reference**

19. [Invariance Properties](#19-invariance-properties)
20. [Dimension Summary Table](#20-dimension-summary-table)
21. [Configuration and Quick-Start](#21-configuration-and-quick-start)
22. [Legacy GATv2Conv Model](#22-legacy-gatv2conv-model)

**Part IV-B — Learning Rate and Diagnostics**

23. [Cosine Learning Rate Scheduler](#23-cosine-learning-rate-scheduler)
24. [Convergence Diagnostic — Backbone vs Heads](#24-convergence-diagnostic--backbone-vs-heads)

**Part V — Extension Notes**

25. [Custom Tile Counts (N > 90 Tiles)](#25-custom-tile-counts-n--90-tiles)
26. [Extending to N > 2 Players](#26-extending-to-n--2-players)
27. [Training Time Estimates](#27-training-time-estimates-nvidia-a6000-48-gb--20-cpu-cores-512-gb-ram)

---

# Part I — Game Engine

## 1. Game Overview

Carcassonne is a tile-placement board game for 2 players.  Players take turns
drawing a tile from a shuffled deck and placing it adjacent to existing tiles
on the board, matching terrain at shared boundaries.  After placing a tile, the
player may optionally place a meeple (follower) on the newly placed tile to
claim a feature (city, road, farm, or chapel).

The game ends when the deck is exhausted.  Players score points for completed
and uncompleted features.  The player with the highest score wins.

This implementation supports:

- **Base Game** — 24 unique tile types, standard rules
- **Inns & Cathedrals expansion** — 26 additional tile types, big meeples,
  inns (on roads), and cathedrals (in cities)
- **Supplementary rules** — Farmers (meeples on grass), Abbots (on chapels/flowers)

### Key source files

| File | Responsibility |
|------|----------------|
| `carcassonne/carcassonne_game_state.py` | Core game state (board, deck, scores, meeples) |
| `carcassonne/objects/tile.py` | Tile data structure (terrains, connections, rotation) |
| `carcassonne/utils/action_util.py` | Enumerates legal actions for current phase |
| `carcassonne/utils/tile_position_finder.py` | Finds valid tile placements |
| `carcassonne/utils/tile_fitter.py` | Checks if a tile fits at a position |
| `carcassonne/utils/state_updater.py` | Applies actions to produce new state |
| `carcassonne/utils/points_collector.py` | Scores completed/final features |
| `carcassonne/utils/city_util.py` | BFS over city segments |
| `carcassonne/utils/road_util.py` | BFS over road segments |
| `carcassonne/utils/farm_util.py` | BFS over farm (grass) regions |
| `carcassonne/utils/meeple_util.py` | Meeple placement/removal |
| `carcassonne/utils/possible_move_finder.py` | Enumerates meeple placements |
| `games/carcassonne_game_state.py` | Wrapper for training/MCTS interface |
| `games/carcassonne_game.py` | Game factory (`get_initial_state()`) |

---

## 2. Tile Structure

Each tile is a `Tile` object (`carcassonne/objects/tile.py`) with the
following attributes:

| Attribute | Type | Description |
|-----------|------|-------------|
| `description` | `str` | Unique tile type name (e.g. `"city_top"`, `"crossroads"`) |
| `turns` | `int` | Current rotation (0 = original, 1 = 90° CW, 2 = 180°, 3 = 270°) |
| `city` | `[[Side]]` | City groups — each group is a list of connected sides |
| `road` | `[Connection]` | Road segments (pairs of sides that a road connects) |
| `grass` | `[Side]` | Grass sides |
| `farms` | `[FarmerConnection]` | Farmer-placeable regions with adjacency info |
| `shield` | `bool` | City shield (extra point per tile in city) |
| `chapel` | `bool` | Chapel in center |
| `flowers` | `bool` | Flowers in center |
| `inn` | `[Side]` | Inn markers on road segments (I&C expansion) |
| `cathedral` | `bool` | Cathedral in city (I&C expansion) |

### Terrain types

Each of the 4 cardinal sides (TOP, RIGHT, BOTTOM, LEFT) has a terrain type:

| TerrainType | Description |
|-------------|-------------|
| `CITY` | Walled city |
| `ROAD` | Road |
| `GRASS` | Open field |
| `CHAPEL` | Chapel/flowers (center only) |

`tile.get_type(side)` returns the terrain at a given side.

### Rotation

`tile.turn(k)` returns a new Tile rotated `k` × 90° clockwise.  The
`description` is preserved (rotation-invariant), but all side references
(city groups, road connections, grass sides, farms) are remapped.

### Internal connectivity

This is the critical distinction between tiles that share the same side
terrains.  For example:

- A tile with CITY on TOP and RIGHT where the two city sides are **one
  connected** city group: `city = [[TOP, RIGHT]]`
- A tile with CITY on TOP and RIGHT where they are **two separate** city
  groups: `city = [[TOP], [RIGHT]]`

Similarly, a straight road (`road = [Connection(TOP, BOTTOM)]`) differs from a
crossroads where roads terminate at the center.  The tile's `description` string
uniquely identifies these structural differences.

---

## 3. Board Representation

The board is a **dictionary** mapping `(row, col)` tuples to `Tile` objects:

```python
board: Dict[Tuple[int, int], Tile]
```

This replaced an earlier fixed-size 2D array and has two key advantages:

- **No size limit** — the board grows unbounded in all directions
- **O(1) lookups** — `board.get((r, c))` to check if a position is occupied

The first tile is placed at `Coordinate(0, 0)`.  Subsequent tiles extend
outward from there with no artificial boundary.

### Coordinate system

- `row` increases downward (TOP neighbour has `row - 1`)
- `column` increases rightward (RIGHT neighbour has `col + 1`)

The four cardinal direction deltas are:

```
TOP:    (-1,  0)
RIGHT:  ( 0, +1)
BOTTOM: (+1,  0)
LEFT:   ( 0, -1)
```

---

## 4. Game Phases and Turn Flow

Each turn has two phases:

```
┌─────────────────────────────────────────────────────────────────┐
│  TILES phase                                                    │
│  Player draws next_tile and places it (TileAction)              │
│  If no valid placement exists → PassAction (discard tile)       │
│                        ↓                                        │
│  MEEPLES phase                                                  │
│  Player may place a meeple on the just-placed tile (MeepleAction)│
│  Or pass (PassAction)                                           │
│  Or remove an abbot (MeepleAction with remove=True)             │
│                        ↓                                        │
│  Score completed features, draw next tile, advance player       │
└─────────────────────────────────────────────────────────────────┘
```

The `GamePhase` enum has two values: `TILES` and `MEEPLES`.

### Action types

| Class | Phase | Fields | Description |
|-------|-------|--------|-------------|
| `TileAction` | TILES | `tile`, `coordinate`, `tile_rotations` | Place tile at position with rotation |
| `MeepleAction` | MEEPLES | `meeple_type`, `coordinate_with_side`, `remove` | Place or remove a meeple |
| `PassAction` | Either | (none) | Skip placement / discard tile |

### Meeple types

| MeepleType | Value | Count per player | Placement |
|------------|-------|------------------|-----------|
| NORMAL | 0 | 7 | On city/road/chapel side |
| ABBOT | 1 | 1 (if ABBOTS rule) | On chapel/flowers center |
| FARMER | 2 | Uses normal pool | On grass (farm) side |
| BIG | 3 | 1 (if I&C expansion) | Like normal but counts as 2 |
| BIG_FARMER | 4 | Uses big pool | Like farmer but counts as 2 |

---

## 5. Legal Move Generation

Legal moves are computed by `ActionUtil.get_possible_actions(game_state)`:

### TILES phase — finding valid tile placements

`TilePositionFinder.possible_playing_positions(game_state, tile_to_play)`:

1. **Empty board:** Return the single starting position `Coordinate(0, 0)`.
2. **Non-empty board:** Iterate over all placed tiles in `board.keys()`.  For
   each placed tile, check all 4 cardinal neighbours.  If a neighbour position
   is empty, it becomes a candidate slot.
3. **For each candidate slot:** Try all 4 rotations of the tile.  For each
   rotation, call `TileFitter.fits()` to check if the tile matches all
   existing neighbours at that slot.
4. **Return** a list of `PlayingPosition(coordinate, turns)` for every valid
   (slot, rotation) combination, which are converted to `TileAction` objects.

If no valid placements exist (rare), a `PassAction` is returned (the tile is
discarded).

### Tile fitting rules (`TileFitter`)

A tile fits at a position if and only if:

- It has **at least one neighbour** (no isolated placements)
- **City edges match:** If the candidate tile has CITY on a side, the
  neighbouring tile on that side must also have CITY, and vice versa
- **Road edges match:** Similarly for ROAD
- **Grass edges match:** GRASS must face GRASS

```
Valid:          Invalid:
┌──────┐        ┌──────┐
│ CITY │        │ CITY │
└──┬───┘        └──┬───┘
   │  CITY         │  GRASS    ← mismatch!
┌──┴───┐        ┌──┴───┐
│ CITY │        │GRASS │
└──────┘        └──────┘
```

### MEEPLES phase — finding valid meeple placements

`PossibleMoveFinder.possible_meeple_actions(game_state)` examines the
just-placed tile and generates meeple options:

1. **City sides:** For each city group on the tile, check via
   `CityUtil.find_city()` (BFS) whether the connected city already contains a
   meeple.  If not, generate a `MeepleAction` for NORMAL, BIG, or ABBOT placement.
2. **Road sides:** Same logic with `RoadUtil.find_road()`.
3. **Chapel/Flowers center:** If the tile has a chapel or flowers, allow
   placing a NORMAL meeple or ABBOT on the CENTER side.
4. **Farm sides:** If the FARMERS supplementary rule is active, check each
   farm region via `FarmUtil.find_farm()`.  If unoccupied, allow FARMER or
   BIG_FARMER placement.
5. **Abbot removal:** If the player has an abbot on the board, add a
   `MeepleAction(remove=True)` to recall it (scoring its chapel/flowers).
6. **PassAction** is always available (skip meeple placement).

All meeple placements are subject to the player's remaining meeple counts
(7 normal, 1 abbot, 1 big meeple).

### Graph-based legal move generation (for the GNN agent)

The GNN agent has a parallel legal-move generator
(`get_graph_legal_tile_placements`) that works purely from the graph structure:

1. **BFS from node 0** using direction-aware edge features to compute relative
   `(row, col)` positions for every node.
2. **Find empty slots:** For each node and each cardinal direction with no
   outgoing edge, record the adjacent empty position as a candidate.
3. **Test fitting:** For each slot, look up its (up to 4) neighbour tiles from
   the relative position map and call `TileFitter.fits()`.
4. **Convert** relative positions back to absolute coordinates using the
   origin node's position.

This is topologically equivalent to the standard `TilePositionFinder` but
operates on the graph without scanning a grid.

---

## 6. Scoring Rules

Points are collected by `PointsCollector` in two situations:

### During the game — completed features

After each tile placement (and meeple phase), `remove_meeples_and_collect_points`
checks features touching the newly placed tile:

**Cities:**

- A city is complete when every city edge has a matching city edge on a
  neighbouring tile (no open sides).
- The **winner** is the player with the most meeples in the city (big meeples
  count as 2).  Ties → all tied players score.
- Points: **2 per tile** in the city, **+2 per shield**, **+2 per cathedral
  tile** (I&C).  If the city has a cathedral and is **not** completed by
  end-of-game, it scores **0** instead.
- All meeples in the completed city are returned to their owners.

**Roads:**

- A road is complete when both endpoints terminate (at a city, crossroads, or
  loop back to itself).
- The winner is determined the same way as cities.
- Points: **1 per tile** on the road.  If the road has an inn and is completed:
  **2 per tile**.  If it has an inn and is **not** completed by end-of-game: **0**.
- Meeples are returned.

**Chapels / Flowers:**

- Complete when all 8 surrounding positions (3×3 grid) are occupied.
- Points: **9** (1 for the chapel tile + 8 surrounding).
- Meeple is returned.

### End of game — final scoring

`count_final_scores` is called when the deck is empty:

- **Uncompleted cities:** 1 per tile + 1 per shield.  Cathedral cities score 0.
- **Uncompleted roads:** 1 per tile.  Inn roads score 0.
- **Uncompleted chapels:** 1 per occupied tile in the 3×3 area.
- **Farms (if FARMERS rule):** For each farm region, count the number of
  **completed cities** adjacent to that farm.  The player(s) with the most
  farmer-type meeples in the farm score **3 per completed adjacent city**.

### Determining the winner

The game wrapper (`games/carcassonne_game_state.py`) computes player values:

- If scores differ: winner gets `+1`, loser gets `-1`
- If scores are equal: both get `0` (draw)

---

## 7. State Transitions

`StateUpdater.apply_action(game_state, action)` produces a new state:

1. **Copy** the current state (`simple_copy()` — shallow copy of the board dict).
2. **Apply the action:**
   - `TileAction` → place the tile on the board, set phase to MEEPLES, record
     `last_tile_action`.
   - `MeepleAction` → place or remove meeple, update meeple counts.  If
     removing an abbot, score its chapel/flowers immediately.
   - `PassAction` → in TILES phase, transitions to MEEPLES; in MEEPLES phase,
     no-op.
3. **After MEEPLES phase completes:**
   - `remove_meeples_and_update_score()` — score any features completed by the
     newly placed tile.
   - `draw_tile()` — pop the next tile from the deck (or set `next_tile = None`
     if deck is empty, triggering game termination).
   - `next_player()` — advance to the other player, reset phase to TILES.
4. **If terminated:** `count_final_scores()` — score all uncompleted features,
   farms.

The state is immutable from the caller's perspective — `apply_action` always
returns a new copy.

---

# Part II — Neural Architecture

## 8. Why a Graph Transformer?

The board state is represented as a graph: one node per placed tile, edges
between orthogonally adjacent tiles.  This is inherently
**translation-invariant** — shifting the entire board changes no features.

However, a standard GNN (like the legacy 4-layer GATv2Conv model) has a
limited receptive field — information can only travel as many hops as there
are layers.  In Carcassonne, a tile at (0,0) may need to reason about a tile
many hops away (e.g. to evaluate whether placing at a position completes a
distant city).

The **GPS (General, Powerful, Scalable) Graph Transformer** solves this by
combining:

- **Local message passing** (GATv2Conv) for fine-grained neighbour interactions
- **Global multi-head self-attention** so every tile can attend to every other
  tile in a single layer
- **RWPE** for global structural awareness without any coordinate dependence

Additionally, the GPS model uses carefully designed **rotation-invariant** and
**chirality-preserving** features, so the network need not learn these
symmetries from data.

```
Legacy GNN                      GPS Graph Transformer
──────────────                  ──────────────────────────────
4× GATv2Conv layers             8× GPS layers (GATv2Conv + MHSA + FFN)
104-dim node features           100-dim node features + 32-dim RWPE
  (rotation-variant)              (rotation-invariant)
7-dim edge features             3-dim edge features for model
  (includes direction bits)       (terrain only, no direction)
4-hop receptive field           Global context via self-attention
~243K parameters                ~3.32M parameters
```

---

## 9. Tile Vocabulary and Node/Edge Creation

### Tile vocabulary

Every tile has a description string.  The sorted union of Base Game and
Inns & Cathedrals decks gives 50 unique types:

```python
TILE_DESCRIPTIONS = sorted(base_tiles.keys() | inns_and_cathedrals_tiles.keys())
# 50 entries: 'bent_road', 'bent_road_flowers', 'chapel', ...
```

Each maps to an integer index 0–49.  Both the legacy and GPS models use this
as a 50-dim type one-hot.  Crucially, `tile.description` does not change when
a tile is rotated (`tile.turn()` preserves the description), so the type
one-hot is **rotation-invariant** and is included in both models.

### Nodes

Each placed tile on the board becomes exactly one node.  Nodes are ordered by
`(row, col)` for determinism.

### Edges

For each pair of orthogonally adjacent tiles, two directed edges are created
(A→B and B→A).  Each edge carries a 7-dim feature vector (see
[Section 12](#12-edge-feature-split)).

---

## 10. Invariant Feature Encoding (GPS)

### Why both tile type AND canonical terrain?

The GPS node features include **two** tile identity encodings that serve
complementary roles:

| Feature | Dims | What it captures | Rotation-invariant? |
|---------|------|-----------------|---------------------|
| Tile type one-hot | 50 | Internal connectivity (connected vs disconnected cities, straight roads vs crossroads) | Yes (`tile.description` unchanged by `turn()`) |
| Canonical terrain sequence | 20 | What terrain is on each side, in canonical order | Yes (cyclic-shift minimum) |

**Why is the type one-hot needed?**  Two tiles can have identical terrain on
all four cardinal sides but differ in *internal connectivity*:

```
Tile A: "city_top_right" (connected)       Tile B: "city_top_right_disconnected"
  ┌─────────┐                                ┌─────────┐
  │ CITY▓▓▓▓│                                │ CITY  ▓▓│
  │ ▓▓▓▓▓▓▓▓│                                │       ▓▓│
  │ ▓▓▓▓▓▓▓▓│  CITY                          │       ▓▓│  CITY
  │ grass   │                                │ grass   │
  └─────────┘                                └─────────┘

Both have TOP=CITY, RIGHT=CITY, BOTTOM=GRASS, LEFT=GRASS
→ identical canonical terrain sequence!
But: one has a single connected city, the other has two separate cities.
```

The same issue arises with roads (straight road vs crossroads with the same
side terrains).  The tile type one-hot resolves this because each unique
description corresponds to a unique internal structure.

**Why is the canonical terrain sequence still needed?**  It provides explicit,
structured terrain-layout information that the model can use directly for
terrain-matching reasoning, without having to learn the mapping from type
index to terrain from scratch.

### Canonical terrain sequence (20-dim)

**Algorithm:**

1. Extract the terrain type at each cardinal side in clockwise order:
   `[TOP, RIGHT, BOTTOM, LEFT]`, each as a 5-dim one-hot → 20 values total.
2. Consider all 4 cyclic shifts (rotations) of this 20-dim vector.
3. Pick the **lexicographically smallest** shift as the canonical form.

**Worked example — `city_top_straight_road` at rotation 0:**

```
Terrain at sides: TOP=CITY, RIGHT=GRASS, BOTTOM=ROAD, LEFT=GRASS
One-hots:         [1,0,0,0,0] [0,0,1,0,0] [0,1,0,0,0] [0,0,1,0,0]

4 cyclic shifts:
  shift 0: [1,0,0,0,0, 0,0,1,0,0, 0,1,0,0,0, 0,0,1,0,0]  ← CITY,GRASS,ROAD,GRASS
  shift 1: [0,0,1,0,0, 0,1,0,0,0, 0,0,1,0,0, 1,0,0,0,0]  ← GRASS,ROAD,GRASS,CITY
  shift 2: [0,1,0,0,0, 0,0,1,0,0, 1,0,0,0,0, 0,0,1,0,0]  ← ROAD,GRASS,CITY,GRASS
  shift 3: [0,0,1,0,0, 1,0,0,0,0, 0,0,1,0,0, 0,1,0,0,0]  ← GRASS,CITY,GRASS,ROAD

Lexicographic minimum: shift 2 → [0,1,0,0,0, 0,0,1,0,0, 1,0,0,0,0, 0,0,1,0,0]
```

Rotate the tile 90° CW.  The sides become `TOP=GRASS, RIGHT=CITY,
BOTTOM=GRASS, LEFT=ROAD`.  The same 4 cyclic shifts produce the same set of
candidates → same canonical form.

**Chirality preservation:** We only consider cyclic shifts, **not** reflections.
Two tiles that are mirror images (e.g. `CITY,ROAD,GRASS,GRASS` clockwise vs
`CITY,GRASS,GRASS,ROAD` clockwise) produce **different** canonical sequences.
This is important because mirrored tiles have different gameplay consequences.

### GPS tile static features (74-dim)

```
[0 :50]  Tile type one-hot (rotation-invariant)
[50:70]  Canonical terrain sequence (rotation-invariant, chirality-preserving)
[70:74]  Tile flags: chapel, shield, cathedral, flowers
```

### GPS node feature vector (100-dim)

```
Indices     Section                         Dims
───────────────────────────────────────────────────
[0  :74]    GPS tile static                 74
[74]        Just-placed flag                 1
[75]        Has-meeple flag                  1
[76 :79]    Meeple owner one-hot             3
[79 :84]    Meeple type one-hot              5
[84 :93]    Meeple side one-hot              9
[93 :100]   Game globals                     7
───────────────────────────────────────────────────
Total                                       100
```

Game globals (7 values): player meeple count / 7, player abbot count,
player big meeple count, enemy meeple count / 7, enemy abbot count,
enemy big meeple count, is-meeples-phase flag.

RWPE (32-dim) is stored separately and injected via a learned encoder.

---

## 11. Random Walk Positional Encoding (RWPE)

RWPE provides global structural awareness without any coordinate dependence.

**Definition:** For each node *i*, compute the diagonal entries of the random
walk operator raised to powers 1 through *K*:

```
RWPE_i = [RW¹_ii, RW²_ii, ..., RW^K_ii]
```

where `RW = A·D⁻¹` is the transition matrix (D = degree matrix).  Entry
`RW^k_ii` is the probability of a random walk starting at node *i* returning
to *i* after exactly *k* steps.

**Properties:**

- Depends **only on graph topology** → translation-invariant and
  rotation-invariant
- No sign ambiguity (unlike Laplacian eigenvectors)
- Captures local structure (low k: degree, triangle count) and global
  structure (high k: community membership, graph diameter)

**Implementation:** For graphs ≤500 nodes (all Carcassonne games), we use
dense matrix powers.  For larger graphs, sparse operations are used.

**Worked example — 3-tile line graph (0—1—2):**

```
Degree: node 0=1, node 1=2, node 2=1

RW¹₀₀ = 0     (walk from 0, must go to 1)
RW¹₁₁ = 0     (walk from 1, goes to 0 or 2)

RW²₀₀ = 0.5   (0→1 with prob 1, then 1→0 with prob 0.5)
RW²₁₁ = 1.0   (1→0 then 0→1, OR 1→2 then 2→1 — both certain)
RW²₂₂ = 0.5   (symmetric with node 0)
```

Nodes 0 and 2 have identical RWPE (they are structurally equivalent).

**In the model:** RWPE is processed by a small MLP
(`Linear(32→128) → GELU → Linear(128→128)`) and **added** to the projected
node features before the GPS layers.

---

## 12. Edge Feature Split

Each edge carries a 7-dim feature:

```
[0:3]  Terrain one-hot: [CITY, ROAD, GRASS]
[3:7]  Direction one-hot: [d_TOP, d_RIGHT, d_BOTTOM, d_LEFT]
```

These serve **two different purposes**:

| Attribute | Dims | Used by |
|-----------|------|---------|
| `edge_attr` (7-dim) | 3 terrain + 4 direction | BFS for legal move generation |
| `edge_attr_model` (3-dim) | 3 terrain only | GPS model input |

The direction bits are essential for BFS-based relative position mapping (used
to find legal tile placements), but they encode spatial orientation and would
break rotation invariance if fed to the model.  The GPS model therefore sees
**only the 3-dim terrain features**.

---

## 13. GPS Graph Transformer Architecture

```
Input:
  x_gps          (N, 100)   GPS node features
  rwpe           (N, 32)    Random walk positional encoding
  edge_attr_model (E, 3)    Terrain-only edges

Node projection:  Linear(100 → 128)
RWPE encoder:     Linear(32 → 128) → GELU → Linear(128 → 128)
Edge encoder:     Linear(3 → 128)

x = node_proj(x_gps) + rwpe_enc(rwpe)      →  (N, 128)
edge_h = edge_enc(edge_attr_model)          →  (E, 128)

8× GPSLayer:
  ├─ Local:   GATv2Conv(128, 128, heads=4, concat=False, edge_dim=128) + residual + LayerNorm
  ├─ Global:  MultiheadAttention(128, 4 heads, batch_first=True)       + residual + LayerNorm
  └─ FFN:     Linear(128→512) → GELU → Linear(512→128)                + residual + LayerNorm

Output: node_embeddings  (N, 128)
```

### GPSLayer details

Each layer has three sub-layers, all with residual connections and LayerNorm:

1. **Local message passing (GATv2Conv):** 4-head graph attention with edge
   features.  Captures fine-grained interactions between directly adjacent
   tiles (terrain matching, city/road continuation).

2. **Global self-attention:** Standard multi-head attention over all nodes
   within each graph (using `data.batch` to separate graphs in a batch).
   Every tile can attend to every other tile regardless of graph distance.

3. **Feed-forward network:** Position-wise MLP with GELU activation and 4×
   expansion factor.

The combination gives each GPS layer both **local** and **global** context,
unlike pure GNN layers which only see immediate neighbours.

---

## 14. Value Head

```
node_embeddings  (N, 128)
       ↓ global_mean_pool
graph_embedding  (1, 128)
       ↓ Linear(128 → 64) + LeakyReLU
       ↓ Linear(64 → 1) + tanh
value  ∈ [-1, +1]
```

The tanh squashes output to match AlphaZero training targets (+1 = win,
-1 = loss, 0 = draw).

---

## 15. Policy Head — Action Scorer

Three per-action-type scorers, each producing a scalar score per legal action.
Only legal actions are ever scored — no fixed action space.

### Tile placement scorer

Input: 205-dim vector (GPS model) or 213-dim (legacy):

```
GPS model (205-dim):
  [0  : 74]   GPS tile static features (type one-hot + canonical terrain + flags)
  [74 :202]   Mean of GNN embeddings of existing neighbours (128-dim)
  [202:205]   Sum of terrain one-hots across neighbours (3-dim)

Linear(205 → 256) + LeakyReLU → Linear(256 → 128) + LeakyReLU → Linear(128 → 1)
```

Neighbour lookup uses BFS-based relative position mapping from the graph's
direction-aware edges — no coordinate tensors.

### Meeple placement scorer

Input: 142-dim (same for both models):

```
[0  :128]   GNN embedding of the just-placed tile
[128:133]   Meeple type one-hot (5)
[133:142]   Meeple side one-hot (9)

Linear(142 → 128) + LeakyReLU → Linear(128 → 1)
```

### Pass / remove-abbot

A single learnable bias parameter (`self.pass_bias`).

### Score → probability

`softmax` across all K legal action scores produces the policy distribution.

---

# Part III — Training and Search

## 16. AlphaZero Training Pipeline

The training loop in `carcassonne_train.py` follows the standard AlphaZero
algorithm:

### Iteration structure

```
for each iteration (default 100):
    1. Self-play:  100 episodes using MCTS + current model
    2. Train:      Update model on accumulated data
    3. Arena:      (Optional) new model vs previous, reject if worse
    4. Evaluate:   New model vs random agent
    5. Checkpoint: Save model + training data
```

### Self-play episode

```python
def execute_episode():
    state = game.get_initial_state()
    trajectory = []

    while not state.is_terminal():
        temperature = 1 if turn < temperature_threshold else 0
        action_probs = agent.get_action_probs(state, temperature, exploration_noise=True)
        trajectory.append((state, action_probs))

        action = sample(action_probs)
        state = state.apply_action(action)

    # Assign values: winner gets +1, loser gets -1
    for (state, probs) in trajectory:
        value = terminal_value * (-1 if state.player != terminal_player else 1)
        train_data.append((state, probs, value))
```

Key details:

- **Temperature:** For the first `temperature_threshold` moves (default 60),
  `temperature=1` (proportional to visit counts, encouraging exploration).
  After that, `temperature=0` (argmax — best move).
- **Exploration noise:** Dirichlet noise added to the root node's prior
  probabilities during self-play.
- **Value assignment:** The terminal game value is propagated back through the
  trajectory, flipped for the opponent's perspective.

### Training data management

- Each iteration generates up to `max_train_data_per_iter` samples (default
  100 episodes × 180 turns).
- Training data is accumulated across iterations, keeping the most recent
  `max_train_data_iters` iterations (default 4).
- All accumulated data is used for training each iteration.

### Arena evaluation

If `arena_win_threshold > 0`:

- Play `arena_iterations` games (half as player 1, half as player 2).
- If the new model's win rate is below the threshold, reject it and revert.

Regardless, the model is always evaluated against a random agent.

### Checkpointing

Each iteration saves:

- Model weights
- Training state (iteration number)
- All training data (serialized as JSON)

Checkpoints can be resumed with `load_checkpoint_path`.

---

## 17. Monte Carlo Tree Search (MCTS)

MCTS is the core search algorithm that converts neural network evaluations
into action probabilities.

### Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_search` | 25 | Number of MCTS simulations per move |
| `cpuct` | 1.25 | Exploration constant in UCB formula |
| `root_dirichlet_alpha` | 0.25 | Dirichlet noise parameter |
| `root_exploration_fraction` | 0.05 | Fraction of noise to mix in |

### Search algorithm

Each call to `get_action_probs(state, temperature)` runs `num_search`
iterations:

```
for each simulation:
    1. SELECT:    Traverse tree from root, picking child with highest UCB score
    2. EXPAND:    At a leaf node, call the neural network to get (policy, value)
    3. BACKPROP:  Update visit counts and value estimates up the tree
```

**UCB formula:**

```
UCB(s, a) = Q(s, a) + cpuct × P(s, a) × √(N_parent) / (1 + N_child)
```

where Q is the mean value, P is the prior from the neural network, and N is
the visit count.

### Temperature-based action selection

After all simulations complete:

- `temperature = 0` → deterministic: pick the most-visited action
- `temperature > 0` → stochastic: probability proportional to
  `visit_count^(1/temperature)`

### Neural network interface

The MCTS calls `inference_fn(state, legal_actions)` which:

1. Builds a PyG graph from the game state
2. Runs the GPS model forward pass
3. Returns `{action: probability}` policy and scalar value

---

## 18. Variable-K Training Loop

The GNN agent's training handles variable numbers of legal actions per state:

1. **Batch the GNN backbone:** `Batch.from_data_list` treats B graphs as one
   large disconnected graph.  One forward pass computes all node embeddings.
2. **Per-sample scoring:** For each sample, extract its nodes and score its
   legal actions.
3. **Scatter softmax:** `pyg_softmax` applies softmax within each sample's
   action group — no cross-contamination between samples.
4. **Policy loss:** Cross-entropy between MCTS target policy and model policy:
   `-(1/B) Σ_i Σ_k target_ik × log(prob_ik)`
5. **Value loss:** MSE between predicted value and game outcome:
   `(1/B) Σ_i (v_predicted - v_target)²`
6. **Combined loss:** `policy_loss + value_loss`, with optional gradient clipping.

---

# Part IV — Properties and Reference

## 19. Invariance Properties

### Translation invariance

All features depend only on graph topology (adjacency, terrain types), never
on absolute coordinates.  Shifting every tile by `(Δr, Δc)` changes no node
features, no edge features, and no RWPE values.

**Verified by test:** Translate all coordinates by (+100, -50), rebuild graph,
assert all GPS features and RWPE match exactly.

### Rotation invariance

Four independent mechanisms ensure rotation invariance:

1. **Tile type one-hot:** `tile.description` is unchanged by `tile.turn()`,
   so the 50-dim type one-hot is the same regardless of rotation.
2. **Canonical terrain sequence:** All 4 rotations of the same tile produce
   the same 20-dim feature vector (cyclic shifts → lexicographic minimum).
3. **Model edge features:** Only 3-dim terrain (CITY/ROAD/GRASS), no direction
   bits.  The terrain at a boundary doesn't change when the board is rotated.
4. **RWPE:** Random walk return probabilities depend only on graph structure,
   which is preserved under rotation.

**Verified by test:** Rotate entire board 90° CW (remap coordinates and tile
rotations), rebuild graph, assert GPS features and RWPE match (up to node
reordering).

### Chirality preservation

The canonical terrain sequence considers only cyclic shifts, **not** mirror
reflections.  Two tiles that are mirror images produce different canonical
sequences.

**Example:**

```
Tile A: CITY, ROAD, GRASS, CHAPEL     →  canonical: [0,0,0,1,0, 0,1,0,0,0, ...]
Tile B: CITY, CHAPEL, GRASS, ROAD     →  canonical: [0,0,0,1,0, 0,0,1,0,0, ...]
                                                     (different!)
```

**Verified by test:** Manually construct a fully asymmetric tile and its
reflection, assert their canonical sequences differ.

---

## 20. Dimension Summary Table

### GPS Model (default)

| Symbol | Value | Meaning |
|--------|-------|---------|
| `NUM_TILE_TYPES` | 50 | Unique tile descriptions (rotation-invariant) |
| `CANON_TERRAIN_DIM` | 20 | Canonical terrain sequence |
| `RWPE_DIM` | 32 | Random walk positional encoding steps |
| `GPS_TILE_STATIC_DIM` | 74 | Type one-hot (50) + canon terrain (20) + flags (4) |
| `GPS_NODE_FEAT_DIM` | 100 | Full GPS node feature vector |
| `EDGE_MODEL_DIM` | 3 | Terrain-only edge features for model |
| `GPS_LAYERS` | 8 | Number of GPS transformer layers |
| `GPS_HEADS` | 4 | Attention heads per GPS layer |
| `GNN_HIDDEN` | 128 | Node embedding dimension |
| `GPS_TILE_ACTION_FEAT_DIM` | 205 | Input to GPS tile-placement scorer |
| `GPS_MEEPLE_ACTION_FEAT_DIM` | 142 | Input to meeple-placement scorer |
| Total model parameters | ~3.32M | 8 GPS layers + heads + scorers |

### Legacy Model

| Symbol | Value | Meaning |
|--------|-------|---------|
| `TILE_STATIC_DIM` | 78 | Type (50) + rotation (4) + terrain (20) + flags (4) |
| `NODE_FEAT_DIM` | 104 | Full node feature vector |
| `EDGE_FEAT_DIM` | 7 | Terrain (3) + direction (4) |
| `TILE_ACTION_FEAT_DIM` | 213 | Input to tile-placement scorer |
| `MEEPLE_ACTION_FEAT_DIM` | 142 | Input to meeple-placement scorer |
| Total model parameters | ~243K | 4 GATv2Conv layers + heads + scorers |

---

## 21. Configuration and Quick-Start

### Switching between models

```bash
# GPS Graph Transformer (default)
python carcassonne_train.py with model_type=gps

# Legacy GATv2Conv
python carcassonne_train.py with model_type=legacy
```

Or programmatically:

```python
from agents.carcassonne_gnn_agent import CarcassonneGNNAgent

agent = CarcassonneGNNAgent('my_agent', summary_writer, model_type='gps')    # default
agent = CarcassonneGNNAgent('my_agent', summary_writer, model_type='legacy')  # legacy
```

### Running invariance tests

```bash
python test_invariance.py
```

Runs 36 checks covering rotation invariance, translation invariance,
chirality preservation, forward-pass shapes, and RWPE correctness.

### Training configuration

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `model_type` | `'gps'` | `'gps'` or `'legacy'` |
| `num_iters` | 100 | Total training iterations |
| `num_self_play_episodes` | 100 | Self-play games per iteration |
| `temperature_threshold` | 60 | Moves before switching to deterministic play |
| `arena_iterations` | 20 | Games in new-vs-previous arena |
| `arena_win_threshold` | 0 | Win rate to accept new model (0 = always accept) |
| `max_train_data_iters` | 4 | Number of past iterations to keep in training set |
| `train_epochs` | 20 | Training epochs per iteration |
| `train_batch_size` | 128 | Mini-batch size |
| `learning_rate` | 1e-3 | Adam initial learning rate |
| `lr_min` | 1e-5 | Minimum LR for cosine annealing |
| `mcts_config.num_search` | 25 | MCTS simulations per move |

### Requirements

```
torch
torch_geometric
torch_scatter
torch_sparse
tensorboard
Pillow
matplotlib>=3.6.3
tqdm
sacred
numpy
```

---

## 22. Legacy GATv2Conv Model

The original 4-layer GATv2Conv model is preserved as `GNNCarcassonneModel`
for backward compatibility and comparison.  It uses:

- 104-dim node features (including 50-dim type one-hot and 4-dim rotation
  one-hot, which are **not** rotation-invariant)
- 7-dim edge features (including 4-dim direction bits)
- 4 GATv2Conv layers with LayerNorm and LeakyReLU
- 4-hop receptive field (no global attention)

Select it with `model_type='legacy'`.  All training, MCTS, and inference
code paths are shared — only the model class and feature encoding differ.

---

# Part IV-B — Learning Rate and Diagnostics

## 23. Cosine Learning Rate Scheduler

### Motivation

A fixed learning rate throughout training is suboptimal for two reasons:

1. **Early training** benefits from a larger step size to move quickly away
   from random initialization.
2. **Late training** benefits from a smaller step size to fine-tune without
   overshooting the loss basin.

### Implementation

A **cosine annealing** schedule is applied within each training iteration
(i.e. each call to `agent.train()`).  The learning rate decays from
`learning_rate` (default 1e-3) to `lr_min` (default 1e-5) following:

```
lr(epoch) = lr_min + 0.5 × (lr_max - lr_min) × (1 + cos(π × epoch / T))
```

where `T = train_epochs` (default 20).

The schedule **resets at the start of every iteration**.  This is intentional:
each iteration brings a fresh batch of self-play data (potentially from a
different-quality model), so the optimizer should start with a higher learning
rate to adapt quickly, then anneal to consolidate.

This pattern is known as **warm restarts** (Loshchilov & Hutter, 2017).

### Configuration

```bash
python carcassonne_train.py with \
    learning_rate=1e-3 \
    lr_min=1e-5
```

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `learning_rate` | 1e-3 | Peak learning rate at the start of each iteration |
| `lr_min` | 1e-5 | Minimum learning rate at the end of each iteration |

### TensorBoard

The current learning rate is logged as `new_agent_lr` at every global step.
You should see a smooth cosine curve that resets each iteration.

### Parameter groups

The optimizer uses three **parameter groups** (backbone, value_head,
policy_head) sharing the same learning rate schedule.  This makes it easy to
add per-group learning rates later if needed (see
[Section 24](#24-convergence-diagnostic--backbone-vs-heads) for how to
determine whether this is warranted).

---

## 24. Convergence Diagnostic — Backbone vs Heads

### What it measures

At the first and last epoch of each training iteration, a convergence
diagnostic table is printed.  It partitions all model parameters into three
groups:

| Group | Parameters |
|-------|-----------|
| **backbone** | `node_proj`, `rwpe_enc`, `edge_enc`, GPS `layers` |
| **value_head** | `value_mlp` |
| **policy_head** | `tile_scorer`, `meeple_scorer`, `pass_bias` |

For each group, it reports:

| Column | Formula | Meaning |
|--------|---------|---------|
| `#Params` | count of scalar parameters | Group size |
| `\|W\|` | L2 norm of all weights | Current weight magnitude |
| `\|G\|` | L2 norm of all gradients | Raw gradient pressure |
| `\|ΔW\|` | L2 norm of the Adam update step | Effective step taken this batch |
| `\|G\|/\|W\|` | gradient-to-weight ratio | How large the gradient is relative to weight scale |
| `\|ΔW\|/\|W\|` | update-to-weight ratio | How much the weights actually changed |

### Example output

```
[Iter 5 | Epoch 19] Convergence Diagnostic (lr=1.00e-05)
---------------------------------------------------------------
Group           #Params          |W|          |G|          |ΔW|    |G|/|W|   |ΔW|/|W|
-----------------------------------------------------------------------------------------------
backbone         2893440    1.4521e+01    3.2100e-02    8.7000e-04   2.21e-03   5.99e-05
value_head          8385    2.1000e+00    1.5600e-02    6.2000e-04   7.43e-03   2.95e-04
policy_head       107137    3.8400e+00    4.8200e-02    1.4100e-03   1.26e-02   3.67e-04
```

### How to read it

**Healthy training** — all three `|ΔW|/|W|` values are within the same order
of magnitude (e.g. all between 1e-4 and 1e-3).  This means the three
components are converging at roughly the same rate.

**Policy head converging much faster than backbone** — if `|ΔW|/|W|` for
`policy_head` is 10× or more larger than for `backbone`:

- The policy head is moving fast but may be overfitting to stale backbone
  features.
- The backbone is underfitting — not extracting good enough representations.
- **Action:** Lower the policy head LR (use per-group rates) or increase
  backbone capacity/LR.

**Backbone converging much faster than heads** — if `|ΔW|/|W|` for
`backbone` is 10× larger than the heads:

- The representation is changing rapidly but the heads can't keep up.
- **Action:** Reduce backbone LR or increase head LR.

**Value head much smaller `|ΔW|/|W|` than policy head** — common because the
value loss MSE often produces smaller gradients than the cross-entropy policy
loss.  This is generally fine unless the value head's loss plateaus while
policy loss still drops.

**`|G|/|W|` ≫ `|ΔW|/|W|`** — Adam's adaptive scaling is heavily damping
the raw gradient.  Normal when gradients are noisy; the optimizer is doing
its job.  If `|G|/|W|` and `|ΔW|/|W|` are nearly equal, Adam is barely
modifying the step — happens when all gradients are similar magnitude.

### TensorBoard scalars

Per-group ratios are also logged to TensorBoard:

- `new_agent_backbone_grad_to_weight`
- `new_agent_backbone_update_to_weight`
- `new_agent_value_head_grad_to_weight`
- `new_agent_value_head_update_to_weight`
- `new_agent_policy_head_grad_to_weight`
- `new_agent_policy_head_update_to_weight`

Plot these over iterations to see if any group's convergence rate is
diverging from the others.  Stable parallel lines = healthy.  Lines
diverging by >10× over many iterations = intervention needed.

### When to act

| Symptom | Likely cause | Remedy |
|---------|-------------|--------|
| `policy |ΔW|/|W|` >> `backbone |ΔW|/|W|` | Backbone too slow | Increase `learning_rate` or backbone capacity |
| `backbone |ΔW|/|W|` >> heads | Representation churning | Lower `learning_rate` or add warmup |
| All `|ΔW|/|W|` < 1e-6 | LR too small or loss flat | Increase `learning_rate`, check data quality |
| All `|ΔW|/|W|` > 1e-1 | LR too large, risk of divergence | Decrease `learning_rate` |
| `value_head` loss plateaus while policy improves | Value signal is weak | Increase `margin_scale` or train more epochs |

---

# Part V — Extension Notes

## 25. Custom Tile Counts (N > 90 Tiles)

The system can be extended to games with more than 90 tiles by changing tile
proportions while keeping the same 50 tile types.

### What needs to change

**Tile counts dictionary.**  The single source of truth for how many copies of
each tile type enter the deck is `base_tile_counts` in
`carcassonne/tile_sets/base_deck.py` (and `inns_and_cathedrals_tile_counts`
for the expansion).  A custom counts dict with different integers is all that
is needed to change deck composition.

**`CarcassonneGameState.initialize_deck()`** (line ~186) consumes these dicts:

```python
for card_name, count in base_tile_counts.items():
    for i in range(count):
        new_tiles.append(base_tiles[card_name])
```

The suggested approach is to add an optional `tile_counts: Dict[str, int]`
parameter to `CarcassonneGameState.__init__()`.  If provided,
`initialize_deck()` uses it instead of the default counts.  This keeps the
change backward-compatible — pass nothing and you get the standard game.

### What does NOT need to change

| Component | Why it's already fine |
|-----------|----------------------|
| Board representation | `Dict[Tuple[int,int], Tile]` — unbounded, no grid limit |
| Feature completion (CityUtil, RoadUtil, PointsCollector) | BFS-based, operates on whatever tiles are on the board |
| Legal move generation (TileFitter, ActionUtil) | Iterates empty neighbours of occupied cells; scales with board size |
| GNN / Graph Transformer | Variable-size graph; `Batch.from_data_list` handles any N |
| Tile-type one-hot (50-dim) | Based on `tile.description` — same 50 types regardless of copy count |
| RWPE | Computed from adjacency matrix of whatever graph is present |

### Training considerations

- Longer games → more MCTS simulations per episode, longer training time.
- RWPE dimension (32) may benefit from tuning for larger graphs — increasing
  `RWPE_DIM` captures structure at longer random-walk distances.
- The game-global feature `deck_remaining / total_deck_size` normalizes
  correctly regardless of N.

---

## 26. Extending to N > 2 Players

The game engine is already largely N-player capable, but several layers above
it assume exactly 2 players.  Here is a layer-by-layer breakdown.

### Layer 1 — Game engine (`carcassonne/` package): Mostly ready

The core engine is parameterized by `players=N`:

- `CarcassonneGameState.__init__()` accepts `players: int` and creates
  `meeples`, `abbots`, `big_meeples`, `placed_meeples`, `scores` as lists
  of length `players`.
- `StateUpdater.next_player()` does
  `current_player = (current_player + 1) % players` — wraps for any N.
- `StateUpdater.play_meeple()` indexes by `current_player` — works for any N.
- `PointsCollector` scoring uses player-indexed lists — works for N players.
- An existing 4-player example lives at
  `carcassonne/examples/four_player_game_random_moves.py`.

### Layer 2 — Game state wrapper (`games/carcassonne_game_state.py`): Hardcoded to 2

Two critical issues:

```python
assert lib_state.players == 2            # line 15 — hard assert

def get_player_value(self, player):
    diff = scores[player] - scores[1 - player]   # line 31 — only works for 2
```

The `1 - player` trick only works for 2 players.  For N players, a different
value function is needed.  Options:

| Strategy | Formula | Pros | Cons |
|----------|---------|------|------|
| Relative gap | `(my_score - max_opponent) / normalizer` | Simple, competitive | Ignores non-leading opponents |
| Score fraction | `my_score / sum(all_scores)` | Smooth, cooperative-aware | Degenerates early when scores are 0 |
| Rank-based | +1 for 1st, -1 for last, linear interpolation | Clear training signal | Ignores score magnitude |

### Layer 3 — MCTS (`agents/mcts.py`): Hardcoded to 2-player zero-sum

```python
assert state.get_num_players() == 2  # line 208
```

The value backpropagation assumes 2-player zero-sum:

```python
if parent.current_player != child.current_player:
    value = -value   # simple negation — only valid for 2 players
```

For N players, the MCTS must switch from scalar values to **per-player value
vectors**.  Each node stores `[v0, v1, ..., v_{N-1}]` instead of a scalar.
During selection, each player maximizes their own component.  This is a
well-known extension (multiplayer MCTS / max^n / paranoid search).

### Layer 4 — GNN node features (`agents/carcassonne_gnn_agent.py`): Hardcoded to 2

Several spots assume exactly 2 players:

| Feature | Current | Change needed |
|---------|---------|---------------|
| `MEEPLE_OWNER_DIM` | 3 (none/P0/P1) | N+1 (none + N players) |
| Meeple owner encoding (lines 290–292) | `if owner == 0 ... elif owner == 1` | Extend to 0..N-1 |
| Game globals (lines 360–366) | `player_meeples` + `enemy_meeples` (6 values) | All N players' resources, or "self + avg opponent" canonicalization |
| `for p_idx in range(2)` (line 484) | Iterates 2 players' meeples | Change to `range(state.players)` |

Dimension impacts:

```
MEEPLE_OWNER_DIM:       3 → N+1
Game globals:           7 → 3*N + 1  (3 resource counts per player + phase flag)
GPS_NODE_FEAT_DIM:      100 → 100 + (N+1 - 3) + (3*N + 1 - 7)
                                    = 100 + (N - 2) + (3N - 6)
                                    = 4N + 92
GPS_TILE_ACTION_FEAT_DIM: recompute accordingly
```

For N=4: `GPS_NODE_FEAT_DIM = 108`, `MEEPLE_OWNER_DIM = 5`.

### Layer 5 — Visualizer and training scripts: Cosmetic

- `PLAYER_CLR` in `game_visualizer.py` needs N colors (currently 2).
- Score display, legend, and log messages reference P0/P1 — extend to P0..P(N-1).
- `carcassonne_train.py` prints "player 2 score" — extend for N players.

### Suggested implementation order

1. Remove `assert players == 2` in `games/carcassonne_game_state.py` and
   rewrite `get_player_value()` for N players.
2. Extend MCTS to store per-player value vectors instead of a scalar.  Change
   backpropagation from negation to per-player indexing.
3. Update GNN feature dimensions: `MEEPLE_OWNER_DIM = N+1`, game globals to
   include all players' resources (or canonicalize as "self + opponent summary").
4. Update `GPS_NODE_FEAT_DIM` and `GPS_TILE_ACTION_FEAT_DIM` constants.
5. Update visualizer colors and display.

---

## 27. Training Time Estimates (NVIDIA A6000 48 GB + 20 CPU Cores, 512 GB RAM)

### Model and hardware summary

| Property | Value |
|----------|-------|
| Model parameters | 3,321,092 (3.3M) |
| Architecture | 8 GPS layers, 128 hidden, 4 heads |
| Graph size during play | 1 node (turn 1) → ~72 nodes (end of game) |
| GPU | NVIDIA A6000 (48 GB VRAM, 10752 CUDA cores, 38.7 TFLOPS FP32) |
| CPU | 20 cores |
| RAM | 512 GB |
| VRAM utilization | < 1 GB (model + graphs); VRAM is not a bottleneck |
| RAM per self-play worker | ~400–600 MB (model copy + game state + MCTS tree) |
| RAM for 18 workers | ~8–11 GB; the remaining ~500 GB is ample headroom |

### Observed CPU baseline

From a CPU-only run (no GPU) with default settings (`num_search=25`):

| Phase | Timing |
|-------|--------|
| Single self-play episode | ~340 seconds (5.7 min) |
| Turns per game | ~179 (tile placement + meeple phases) |
| Per MCTS simulation | ~76 ms (network forward + game logic + tree ops) |
| Training (20 epochs, ~360 examples) | ~23 seconds |
| Training (20 epochs, ~720 examples) | ~78 seconds |

### Speedup analysis

#### GPU speedup for single-episode self-play

The 76 ms per MCTS simulation breaks down roughly as:

| Component | CPU time (est.) | GPU time (est.) | Notes |
|-----------|----------------|-----------------|-------|
| Network forward pass | ~50 ms | ~3 ms | 3.3M model, small graph → GPU is massively underutilized |
| Game logic (legal moves, apply action) | ~15 ms | ~15 ms | Pure Python, no GPU benefit |
| MCTS tree operations | ~11 ms | ~11 ms | Pure Python, no GPU benefit |
| **Total per simulation** | **~76 ms** | **~29 ms** | **~2.6× speedup** |

The GPU speedup for a single self-play episode is moderate (~2.6×) because
the Python game engine is the bottleneck, not the neural network.  Training
has better GPU speedup (~5–8×) due to batch parallelism.

#### Parallel CPU self-play (the big win)

The codebase supports **parallel self-play** via the `num_workers` config
parameter.  When `num_workers > 1`, self-play episodes are distributed across
a `multiprocessing.Pool` of CPU-only worker processes (using the `spawn`
start method for CUDA safety).  Each worker holds its own copy of the model
on CPU and runs episodes independently.

**Why CPU workers beat sequential GPU for self-play:**

| Approach | 50 episodes (num_search=100) | Explanation |
|----------|------------------------------|-------------|
| Sequential GPU (1 episode at a time) | 50 × 131s = **~1.8 hours** | 2.6× faster per episode, but still sequential |
| Sequential CPU | 50 × 340s = **~4.7 hours** | Baseline |
| **18 CPU workers in parallel** | ⌈50/18⌉ × 340s = 3 × 340s = **~17 minutes** | **~6× faster than sequential GPU** |

Each worker runs at single-core CPU speed (~340s per episode at
`num_search=25`, scaling linearly with `num_search`).  But with 18 workers
saturating 18 of the 20 cores, wall-clock time divides by 18.  This massively
outperforms sequential GPU because the GPU is idle between simulations anyway.

**Recommended `num_workers`**: Use **18** (= 20 cores minus 2 reserved for OS
and the main training process).  With 512 GB RAM, memory is not a constraint
(18 workers × ~600 MB ≈ 11 GB).

### Recommended training configurations

Three tiers are presented.  "Decently good" means consistently beating random
play and showing clear strategic behavior.  "Good" means strong positional
play.  "Strong" approaches expert human level.

All estimates below assume `num_workers=18` (parallel self-play on CPU) and
GPU for training.

#### Tier 1: Decently good (~beats random 85–95% of the time)

| Parameter | Value |
|-----------|-------|
| `num_search` | 100 |
| `num_iters` | 50 |
| `num_self_play_episodes` | 50 |
| `num_workers` | 18 |
| `train_epochs` | 20 |
| `max_train_data_iters` | 8 |
| `arena_iterations` | 20 |
| `arena_win_threshold` | 0.55 |

**Time breakdown per iteration (with parallel self-play):**

Per-episode wall time at `num_search=100`:
340s × (100/25) = **1360s ≈ 22.7 min/episode**.

| Phase | Calculation | Time |
|-------|-------------|------|
| Self-play (50 episodes, 18 workers) | ⌈50/18⌉ × 1360s = 3 × 1360s | **~68 min** |
| Training (20 epochs, ~72K examples, GPU) | ~560 batches/epoch × 20 × ~50 ms/batch | ~11 min |
| Arena (40 games, sequential GPU) | 40 × 179 turns × 100 sims × 29 ms | ~5.8 hours |
| **Iteration total (with arena)** | | **~7 hours** |
| **Iteration total (no arena)** | | **~1.5 hours** |

| Config variant | Total wall time |
|----------------|----------------|
| With arena gating | 50 × 7 hours ≈ **15 days** |
| **No arena** (`arena_win_threshold=0`) | 50 × 1.5 hours ≈ **3 days** |
| Fast first checkpoint (no arena, 20 iters) | 20 × 1.5 hours ≈ **30 hours** |

Compare to sequential self-play: 17 days without arena → **3 days** with 18
workers.  This is a **5.7× speedup** on the training pipeline.

#### Tier 2: Good (strong positional play, smart meeple usage)

| Parameter | Value |
|-----------|-------|
| `num_search` | 200 |
| `num_iters` | 100 |
| `num_self_play_episodes` | 100 |
| `num_workers` | 18 |
| `arena_win_threshold` | 0.55 |

Per-episode wall time at `num_search=200`:
340s × (200/25) = **2720s ≈ 45.3 min/episode**.

| Phase | Calculation | Time |
|-------|-------------|------|
| Self-play (100 episodes, 18 workers) | ⌈100/18⌉ × 2720s = 6 × 2720s | **~4.5 hours** |
| Training (20 epochs, ~144K examples, GPU) | | ~30 min |
| Arena (40 games, sequential GPU) | 40 × 179 × 200 × 29 ms | ~11.5 hours |
| **Iteration total (with arena)** | | **~16.5 hours** |
| **Iteration total (no arena)** | | **~5.5 hours** |

| Config variant | Total wall time |
|----------------|----------------|
| With arena gating | 100 × 16.5 hours ≈ **69 days** |
| **No arena** | 100 × 5.5 hours ≈ **23 days** |

Compare to sequential self-play: ~100–150 days → **23 days** without arena.

#### Tier 3: Strong (approaching expert human play)

| Parameter | Value |
|-----------|-------|
| `num_search` | 400–800 |
| `num_iters` | 200+ |
| `num_self_play_episodes` | 200+ |
| `num_workers` | 18 |

Per-episode wall time at `num_search=400`:
340s × (400/25) = **5440s ≈ 91 min/episode**.

Self-play per iteration (200 episodes, 18 workers):
⌈200/18⌉ × 5440s = 12 × 5440s ≈ 18.1 hours.

| Config variant | Total wall time |
|----------------|----------------|
| No arena, 200 iters | 200 × ~19 hours ≈ **~160 days** |
| With further optimizations (see below) | **~2–3 months** |

### Additional optimization opportunities

The parallel self-play already captures the largest gain.  Further
improvements stack on top:

| Optimization | Expected speedup | Effort | Status |
|-------------|-----------------|--------|--------|
| **Parallel CPU self-play** (`num_workers=18`) | **~6× on self-play phase** | Done | **Implemented** (set `num_workers=18`) |
| **C/C++ game engine** — rewrite `carcassonne/` in C with Python bindings (game logic is ~60% of per-simulation time) | 2–3× on per-episode time → reduces parallel wall time proportionally | High | Not yet |
| **Increase `num_search` with budget** — 50 sims early, 200+ later | Same wall time, better quality | Trivial config change | Not yet |
| **Reduce `train_epochs` early** — start with 5, increase to 20 as data grows | Saves ~75% training time in early iterations | Trivial | Not yet |
| **Larger batch size** — 128 → 512 or 1024 (48 GB VRAM supports this) | 2–3× on training phase | Low | Not yet |
| **Mixed precision (FP16)** — `torch.cuda.amp` | 1.5–2× on network forward passes | Low | Not yet |
| **Tree reuse across turns** — implement `get_reuse_hash_key()` for Carcassonne (currently returns `None`) | 1.5–2× on MCTS (fewer leaf expansions) | Medium | Not yet |
| **Batched GPU inference server** — centralize leaf evaluations from all workers into a single GPU batch call | 5–10× on forward pass cost; but diminishing returns when game logic dominates | High | Not yet |
| **Parallel arena games** — extend `num_workers` to arena phase | 3–6× on arena phase (currently the bottleneck when arena is enabled) | Medium | Not yet |

**Stacking optimizations**: Parallel self-play (6×) + C/C++ engine (2.5×) +
tree reuse (1.5×) could yield ~22× overall speedup on self-play, bringing
Tier 2 from 23 days to ~4 days and Tier 3 into the 2–3 week range.

### Practical recommendation

Start with Tier 1 parameters with parallel self-play and arena gating
disabled for speed:

```bash
python carcassonne_train.py with \
    num_iters=30 \
    num_self_play_episodes=50 \
    num_workers=18 \
    arena_win_threshold=0 \
    random_arena_iterations=4 \
    margin_scale=40.0
```

This should produce a "decently good" agent in **~2 days** on your hardware.
Monitor `win_rate_against_random` in TensorBoard — once it consistently
exceeds 0.8, the agent has learned basic strategy.  Then increase
`num_search` and `num_self_play_episodes` for further improvement.

### Quick-reference wall-clock estimates

| Goal | Config | Sequential (1 worker) | **Parallel (18 workers)** |
|------|--------|-----------------------|--------------------------|
| Beats random ~90% | 100 sims, 30 iters, 50 ep | ~10 days | **~2 days** |
| Beats random ~98% | 100 sims, 50 iters, 50 ep | ~17 days | **~3 days** |
| Strong positional play | 200 sims, 100 iters, 100 ep | ~100 days | **~23 days** |
| Expert level | 400 sims, 200 iters, 200 ep | ~6 months+ | **~5 months** (+ further opt needed) |

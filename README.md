# Carcassonne AlphaZero

A PyTorch implementation of the **AlphaZero** reinforcement learning algorithm applied to the board game **Carcassonne**, using a **GPS Graph Transformer** neural network and a **C++ game engine** with pybind11 bindings.

Based on:

1. https://github.com/wingedsheep/carcassonne

2. https://github.com/TommyX12/carcassonne-alpha-zero

---

## Table of Contents

1. [Overview](#1-overview)
2. [Project Structure](#2-project-structure)
3. [Game Engine](#3-game-engine)
   - [Game Rules](#31-game-rules)
   - [Tile Structure](#32-tile-structure)
   - [Board Representation](#33-board-representation)
   - [Game Phases and Actions](#34-game-phases-and-actions)
   - [Legal Move Generation](#35-legal-move-generation)
   - [Scoring](#36-scoring)
   - [Game Variants](#37-game-variants)
4. [Neural Network Architecture](#4-neural-network-architecture)
   - [Why a Graph Transformer?](#41-why-a-graph-transformer)
   - [Graph Construction](#42-graph-construction)
   - [Invariant Feature Encoding](#43-invariant-feature-encoding)
   - [GPS Graph Transformer](#44-gps-graph-transformer)
   - [Value Head](#45-value-head)
   - [Policy Head](#46-policy-head)
   - [Dimension Summary](#47-dimension-summary)
5. [Training Algorithm](#5-training-algorithm)
   - [AlphaZero Overview](#51-alphazero-overview)
   - [Monte Carlo Tree Search (MCTS)](#52-monte-carlo-tree-search-mcts)
   - [Multi-Round Batched MCTS](#53-multi-round-batched-mcts)
   - [Cross-Game Batching](#54-cross-game-batching)
   - [Self-Play](#55-self-play)
   - [Training Loop](#56-training-loop)
   - [Arena Evaluation](#57-arena-evaluation)
   - [Replay Buffer and Warm-Up](#58-replay-buffer-and-warm-up)
6. [Performance Optimizations](#6-performance-optimizations)
   - [FP16 Inference](#61-fp16-inference)
   - [Parallel Self-Play](#62-parallel-self-play)
   - [Cross-Game Batching](#63-cross-game-batching)
   - [C++ Game Engine](#64-c-game-engine)
7. [Setup and Installation](#7-setup-and-installation)
8. [Training](#8-training)
9. [Configuration Reference](#9-configuration-reference)
10. [Game Visualization](#10-game-visualization)
11. [Convergence Diagnostics](#11-convergence-diagnostics)
12. [Extending the System](#12-extending-the-system)
    - [Custom Tile Counts](#121-custom-tile-counts)
    - [N > 2 Players](#122-n--2-players)
    - [Replacing the Search Algorithm](#123-replacing-the-search-algorithm)

---

## 1. Overview

This project trains a neural network agent to play 2-player Carcassonne using the AlphaZero algorithm. The system combines:

- **GPS Graph Transformer** — a rotation/translation-invariant graph neural network that operates on the board as a variable-size graph
- **Monte Carlo Tree Search (MCTS)** — multi-round batched search with configurable depth
- **C++ game engine** — fast game logic via pybind11, replacing the original pure-Python engine
- **Parallel self-play** — multiprocessing with cross-game NN batching for GPU utilization
- **FP16 inference** — automatic mixed precision on CUDA devices

The agent learns entirely from self-play, starting from random initialization.

---

## 2. Project Structure

```
carcassonne-alpha-zero-main/
├── carcassonne_train.py           # Main training script (Sacred experiment)
├── self_play_worker.py            # Parallel & cross-game batched self-play
├── game_stats.py                  # Per-game statistics tracking
├── game_visualizer.py             # Matplotlib board visualizer + animation
├── search_depth_analysis.py       # MCTS depth analysis tool
├── setup.py                       # C++ engine build script
├── requirements.txt               # Python dependencies
│
├── agents/                        # Agent implementations
│   ├── carcassonne_gnn_agent.py   #   GPS Graph Transformer + GNN agent
│   ├── mcts.py                    #   MCTS (sequential + multi-round batched)
│   ├── mcts_ml_agent.py           #   MCTS-ML agent base class
│   ├── ml_agent.py                #   ML agent base (save/load/train)
│   ├── agent.py                   #   Abstract agent base
│   └── random_agent.py            #   Uniform random baseline
│
├── games/                         # Game interface wrappers
│   ├── game.py                    #   Abstract Game base class
│   ├── game_state.py              #   Abstract GameState interface
│   ├── carcassonne_game.py        #   Carcassonne game factory
│   └── carcassonne_game_state.py  #   Carcassonne state wrapper
│
├── carcassonne_cpp/               # C++ game engine source
│   ├── CMakeLists.txt
│   ├── src/                       #   Bindings, game_state, tile_sets
│   └── include/carcassonne/       #   Headers for all engine modules
│
├── carcassonne_engine/            # Python package wrapping the C++ .so
│   └── __init__.py
│
├── carcassonne/                   # Pure-Python game engine (fallback)
│   ├── carcassonne_game_state.py  #   Full mutable game state
│   ├── carcassonne_visualiser.py  #   Tkinter + PIL tile renderer
│   ├── objects/                   #   Actions, tile, game_phase, etc.
│   ├── tile_sets/                 #   Base + Inns & Cathedrals decks
│   ├── utils/                     #   Action generation, scoring, BFS
│   └── resources/images/          #   Tile images
│
└── utils/                         # Shared utilities
    ├── utils.py                   #   Checkpointing, dict helpers
    ├── experiment.py              #   Sacred experiment setup
    ├── context.py                 #   Dependency injection
    └── debug.py                   #   Debug flags
```

---

## 3. Game Engine

### 3.1 Game Rules

Carcassonne is a tile-placement board game for 2 players. On each turn:

1. Draw a tile from the deck and **place it** on the board adjacent to existing tiles, matching terrain at shared edges (cities to cities, roads to roads, fields to fields).
2. Optionally **place a meeple** on a feature of the just-placed tile to claim it.
3. When a feature is completed, meeples score points and are returned.
4. At game end, incomplete features and farms are scored. Highest score wins.

### 3.2 Tile Structure

Each tile (`Tile` object) has:

| Attribute | Description |
|-----------|-------------|
| `description` | Unique type name (rotation-invariant, 50 types total) |
| `turns` | Current rotation (0–3, multiples of 90° CW) |
| `city` | City groups — connected side lists |
| `road` | Road segments (endpoint pairs) |
| `grass`/`farms` | Grass sides and farmer-placeable regions |
| `shield` | City shield bonus |
| `chapel`/`flowers` | Center features |
| `inn`/`cathedral` | Inns & Cathedrals expansion markers |

Four terrain types exist per cardinal side: CITY, ROAD, GRASS, CHAPEL. Two tiles can share identical terrain on all sides but differ in internal connectivity (e.g., a connected city vs. two disconnected cities), distinguished by `description`.

### 3.3 Board Representation

The board is a `Dict[Tuple[int, int], Tile]` — unbounded, no grid limit. The first tile is placed at `(0, 0)`.

### 3.4 Game Phases and Actions

Each turn has two phases:

| Phase | Actions |
|-------|---------|
| **TILES** | `TileAction(tile, coordinate, rotation)` or `PassAction` (discard) |
| **MEEPLES** | `MeepleAction(type, side, remove?)` or `PassAction` (skip) |

Meeple types: NORMAL (7 per player), ABBOT (1), FARMER (uses normal pool), BIG (1, I&C), BIG_FARMER (uses big pool).

### 3.5 Legal Move Generation

- **Tile phase**: For each empty position adjacent to a placed tile, test all 4 rotations of the current tile via `TileFitter.fits()`.
- **Meeple phase**: BFS over city/road/farm features from the just-placed tile to check for existing meeple conflicts.

### 3.6 Scoring

**During game** — completed features:
- **Cities**: 2 pts/tile + 2/shield + cathedral bonuses. Majority meeple owner scores.
- **Roads**: 1 pt/tile (2 with inn). Majority owner scores.
- **Chapels**: 9 pts when all 8 surrounding tiles placed.

**End of game** — incomplete features at reduced rates. Cathedral cities and inn roads score 0 if incomplete. Farms score 3 pts per completed adjacent city.

Winner gets value `+1`, loser `-1`, draw `0`.

### 3.7 Game Variants

The `game_variant` config controls which tile set and rules are active:

| Variant | Tiles | Rules |
|---------|-------|-------|
| `"base"` | 24 base types (48 tiles) | Standard scoring, no abbots, no inns/cathedrals |
| `"full"` | Base + 26 I&C types (90 tiles) | Inns, cathedrals, big meeples, abbots |

---

## 4. Neural Network Architecture

All neural network code is in `agents/carcassonne_gnn_agent.py`.

### 4.1 Why a Graph Transformer?

The board is represented as a graph: one node per placed tile, edges between adjacent tiles. This is inherently **translation-invariant**. The GPS Graph Transformer adds:

- **Local message passing** (GATv2Conv) for neighbour interactions
- **Global multi-head self-attention** so every tile attends to every other
- **RWPE** for structural awareness without coordinate dependence
- **Local distance-aware attention bias** for spatial context

### 4.2 Graph Construction

`build_graph_from_state(state)` converts a game state to a PyG `Data` object:

- **Nodes**: One per placed tile, ordered by (row, col)
- **Edges**: Two directed edges per adjacent tile pair (A→B, B→A)
- **Node features**: 100-dim GPS features + 32-dim RWPE (stored separately)
- **Edge features**: 7-dim (3 terrain + 4 direction), but only 3-dim terrain fed to model

### 4.3 Invariant Feature Encoding

**GPS node features (100-dim)**:

| Indices | Feature | Dims |
|---------|---------|------|
| 0–49 | Tile type one-hot (rotation-invariant) | 50 |
| 50–69 | Canonical terrain sequence (cyclic-shift minimum) | 20 |
| 70–73 | Tile flags (chapel, shield, cathedral, flowers) | 4 |
| 74 | Just-placed flag | 1 |
| 75 | Has-meeple flag | 1 |
| 76–78 | Meeple owner one-hot | 3 |
| 79–83 | Meeple type one-hot | 5 |
| 84–92 | Meeple side one-hot | 9 |
| 93–99 | Game globals (meeple counts, phase) | 7 |

The canonical terrain sequence takes the 4-side terrain layout, generates all 4 cyclic rotations, and picks the lexicographically smallest — making it rotation-invariant while preserving chirality.

**RWPE (32-dim)**: Random Walk Positional Encoding — the diagonal of `(A·D⁻¹)^k` for k=1..32. Depends only on graph topology.

### 4.4 GPS Graph Transformer

```
x_gps (N, 100)  →  node_proj(100→128)  ─┐
rwpe  (N, 32)   →  rwpe_enc(32→128)     ─┤→  x = sum  →  (N, 128)
edge_attr (E,3) →  edge_enc(3→128)       │

8× GPSLayer:
  ├─ Local:  GATv2Conv(128→128, 4 heads) + residual + LayerNorm
  ├─ Global: MultiheadAttention(128, 4 heads) with distance bias + residual + LayerNorm
  └─ FFN:    Linear(128→512) → GELU → Linear(512→128) + residual + LayerNorm

Output: node_embeddings (N, 128)
```

The distance-aware attention bias adds learned spatial context within the global self-attention, using bucketed graph distances between nodes.

### 4.5 Value Head

Multi-head attentive readout over node embeddings, followed by an MLP:

```
node_embeddings (N, 128)
    ↓ MultiHeadReadout (4 attention heads, 128→512)
    ↓ Linear(512→64) + LeakyReLU
    ↓ Linear(64→1) + tanh
value ∈ [-1, +1]
```

### 4.6 Policy Head

Three per-action-type scorers, producing scalar scores per legal action:

**Tile placement** (205-dim input → MLP → scalar):
- GPS tile features (74) + mean neighbour embeddings (128) + neighbour terrain sums (3)
- Uses direction-aware neighbour lookup from graph edges

**Meeple placement** (142-dim input → MLP → scalar):
- Just-placed tile embedding (128) + meeple type one-hot (5) + side one-hot (9)

**Pass / remove-abbot**: Learnable bias parameter.

All legal action scores go through `softmax` to produce the policy distribution. Actions are scored in a single batched forward pass.

### 4.7 Dimension Summary

| Symbol | Value | Description |
|--------|-------|-------------|
| `NUM_TILE_TYPES` | 50 | Unique tile descriptions |
| `GPS_NODE_FEAT_DIM` | 100 | Node feature vector |
| `RWPE_DIM` | 32 | Random walk PE steps |
| `GNN_HIDDEN` | 128 | Embedding dimension |
| `GPS_LAYERS` | 8 | Transformer layers |
| `GPS_HEADS` | 4 | Attention heads per layer |
| Total parameters | ~417K (small) | 8 GPS layers + heads + scorers |

---

## 5. Training Algorithm

### 5.1 AlphaZero Overview

AlphaZero trains `f_θ(s) → (p, v)` where `p` is a policy over actions and `v` is a value estimate. The cycle:

1. **Self-play**: MCTS guided by `f_θ` generates games → training data `(state, MCTS_policy, outcome)`
2. **Training**: Fit `f_θ` via cross-entropy (policy) + MSE (value)
3. **Evaluation**: Compare against random agents and optionally previous model

### 5.2 Monte Carlo Tree Search (MCTS)

Each node stores visit count (N), value estimate (Q), prior (P), and children. Selection uses PUCT:

```
PUCT(s, a) = Q(s,a) + cpuct × P(s,a) × √N(parent) / (1 + N(child))
```

Dirichlet noise `Dir(α=0.25)` is mixed into root priors during self-play. Value normalization maps Q to [0,1] using observed min/max.

### 5.3 Multi-Round Batched MCTS

Unlike single-round batched MCTS (which only achieves depth 1), the multi-round approach divides `num_search` simulations into rounds of `batch_size` paths each:

1. Select `batch_size` paths from root to unexpanded leaves
2. Batch-evaluate all leaves with one NN forward pass
3. Expand leaves and backpropagate
4. Repeat — subsequent rounds traverse through newly expanded nodes, building real depth

With `num_search=100` and `batch_size=8`, this achieves average PV depths of 4–6, compared to depth 1 with naive batching.

### 5.4 Cross-Game Batching

When `games_per_worker > 1`, a single worker runs N games concurrently. Each round:

1. All active games select paths via `collect_pending_leaves()`
2. Pending leaves from all games are merged into one large NN batch
3. Results are distributed back via `deliver_results()`

This maximizes GPU utilization without affecting search depth (each game's MCTS tree is independent).

### 5.5 Self-Play

Each episode:
1. Initialize game state
2. At each turn: compute temperature `T = max(0, 1 - turn/threshold)` (smooth decay)
3. Run MCTS with exploration noise, sample action from visit-count distribution
4. Record `(state, action_probs, player)` 
5. At termination: assign `v = tanh(score_diff / margin_scale)` propagated through history with sign flips

20% of self-play games are played against a random opponent to prevent policy collapse.

### 5.6 Training Loop

```
for each iteration:
    1. Schedule: interpolate num_episodes, num_search, cpuct from init→final
    2. Self-play: parallel workers generate episodes → append to replay buffer
    3. Train: SGD with momentum on mini-batches sampled from replay buffer
    4. Arena (optional): new model vs. previous model
    5. Random arena: new model vs. random agent
    6. Checkpoint: save model + optimizer + training state
```

The optimizer is **SGD with momentum** (lr=0.1, momentum=0.9, weight_decay=1e-4) with step-decay LR scheduling.

### 5.7 Arena Evaluation

When `arena_win_threshold > 0`: play `arena_iterations` games (half each seat), reject update if win rate is below threshold. Always plays `random_arena_iterations` games against a random agent to track absolute strength.

### 5.8 Replay Buffer and Warm-Up

A fixed-size replay buffer (`deque`) stores the most recent training samples. On the first iteration (or when resuming with an empty buffer), a **warm-up phase** plays enough games to fill the buffer before training begins.

---

## 6. Performance Optimizations

### 6.1 FP16 Inference

On CUDA devices, all batched NN inference uses `torch.autocast(dtype=torch.float16)`. Outputs are cast back to float32 for stable policy/value extraction. This roughly halves GPU memory bandwidth and improves inference throughput.

### 6.2 Parallel Self-Play

`num_workers` controls how many worker processes run self-play in parallel via `multiprocessing.Pool` (spawn context for CUDA safety). Each worker holds its own model copy.

### 6.3 Cross-Game Batching

When `games_per_worker > 1`, episodes are grouped into chunks and each worker runs multiple games concurrently, merging their MCTS leaf evaluations into a single NN call. This increases GPU batch sizes from `~batch_size` to `~games_per_worker × batch_size` per call.

### 6.4 C++ Game Engine

The game logic is implemented in C++ under `carcassonne_cpp/` with pybind11 bindings. Build with:

```bash
pip install -e .
```

Falls back to the pure-Python engine in `carcassonne/` if the C++ module is unavailable.

---

## 7. Setup and Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended for training)
- C++ compiler with C++17 support (for the game engine)

### Installation

```bash
# Clone and enter the project
git clone <your-repo-url>
cd carcassonne-alpha-zero-main

# Install Python dependencies
pip install -r requirements.txt

# Build the C++ game engine
pip install -e .
```

### Dependencies

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

## 8. Training

### Quick start

```bash
python carcassonne_train.py with 'tag="my_run"'
```

### Override parameters

```bash
python carcassonne_train.py with \
    'tag="fast_test"' \
    num_iters=50 \
    num_workers=8 \
    games_per_worker=4 \
    game_variant="base"
```

### Resume from checkpoint

```bash
python carcassonne_train.py with \
    'tag="my_run"' \
    'load_checkpoint_path="experiments/runs/carcassonne/<timestamp>/checkpoints/<checkpoint_dir>"'
```

### Monitor with TensorBoard

```bash
tensorboard --logdir experiments/runs/carcassonne/
```

---

## 9. Configuration Reference

All parameters are in the `@ex.sacred.config` block of `carcassonne_train.py`:

### General

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_iters` | 200 | Total training iterations |
| `model_type` | `'gps'` | `'gps'` (Graph Transformer) or `'legacy'` (GATv2Conv) |
| `model_size` | `'small'` | Model size variant |
| `game_variant` | `'base'` | `'base'` or `'full'` (Inns & Cathedrals) |
| `num_workers` | 20 | Parallel self-play worker processes |
| `temperature_threshold` | 120 | Turns over which temperature decays from 1→0 |
| `margin_scale` | 10.0 | Denominator in `tanh(score_diff / margin_scale)` |
| `random_opponent_fraction` | 0.2 | Fraction of self-play games vs random |

### Replay Buffer

| Parameter | Default | Description |
|-----------|---------|-------------|
| `replay_buffer_size` | 25000 | Max `(state, policy, value)` tuples kept |

### Optimizer (SGD)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `learning_rate` | 0.1 | SGD initial learning rate |
| `momentum` | 0.9 | SGD momentum |
| `weight_decay` | 1e-4 | L2 regularization |
| `gradient_clipping` | 2.0 | Max gradient norm |
| `lr_milestones` | [100, 150] | Iterations at which LR is decayed |
| `lr_gamma` | 0.5 | LR multiplied by this at each milestone |

### Training

| Parameter | Default | Description |
|-----------|---------|-------------|
| `train_steps_per_iter` | 100 | Gradient steps per iteration |
| `train_batch_size` | 256 | Mini-batch size |

### MCTS

| Parameter | Default | Description |
|-----------|---------|-------------|
| `mcts_batch_size` | 16 | Leaves per round in multi-round batched MCTS |
| `games_per_worker` | 1 | Games per worker (>1 enables cross-game batching) |
| `root_exploration_fraction` | 0.25 | Dirichlet noise fraction at root |

### Scheduled Parameters (linearly interpolated init→final)

| Parameter | Init | Final | Description |
|-----------|------|-------|-------------|
| `num_self_play_episodes` | 40 | 80 | Self-play games per iteration |
| `num_search` | 100 | 200 | MCTS simulations per move |
| `cpuct` | 2.0 | 1.25 | Exploration constant |

### Arena

| Parameter | Default | Description |
|-----------|---------|-------------|
| `arena_iterations` | 0 | Games for new-vs-prev model (0 = disabled) |
| `arena_win_threshold` | 0.0 | Win rate to accept new model |
| `random_arena_iterations` | 50 | Games vs random agent per iteration |

---

## 10. Game Visualization

`game_visualizer.py` renders completed games as board images or MP4 animations:

```bash
# Run a random game and produce an animation
python game_visualizer.py --variant base --output game_animation.mp4
```

`search_depth_analysis.py` measures and plots MCTS depth metrics:

```bash
# Fast random-policy analysis
python search_depth_analysis.py --mode random --num-search 20 50 100 200

# Real NN analysis (slower)
python search_depth_analysis.py --mode nn --num-search 20 50 --batch-size 8 --max-turns 30
```

---

## 11. Convergence Diagnostics

At the first and last training step of each iteration, a diagnostic table is printed:

```
[Iter 5 | Step 99] Convergence Diagnostic (lr=5.00e-02)
Group           #Params     |W|          |G|          |ΔW|       |G|/|W|     |ΔW|/|W|
backbone         313568   5.00e+01    6.76e-03    3.38e-04    1.35e-04    6.75e-06
value_head        20929   1.67e+01    2.82e-02    1.41e-03    1.68e-03    8.42e-05
policy_head       83203   1.31e+01    2.28e-03    1.14e-04    1.74e-04    8.69e-06
```

**Key indicators**:
- `|ΔW|/|W|` should be in the same order of magnitude across groups (1e-4 to 1e-3)
- If policy head converges 10× faster than backbone → backbone is underfitting
- If all `|ΔW|/|W|` < 1e-6 → LR too small or loss is flat
- If all `|ΔW|/|W|` > 1e-1 → LR too large, risk of divergence

These ratios are also logged to TensorBoard as `new_agent_<group>_grad_to_weight` and `new_agent_<group>_update_to_weight`.

---

## 12. Extending the System

### 12.1 Custom Tile Counts

The tile counts are defined in `carcassonne/tile_sets/base_deck.py` and `inns_and_cathedrals_deck.py`. Change these dictionaries to alter deck composition. Everything else (board, scoring, GNN, MCTS) scales automatically.

### 12.2 N > 2 Players

The game engine supports N players internally. To extend the full system:

1. **Game wrapper** (`games/carcassonne_game_state.py`): Remove `assert players == 2`, rewrite `get_player_value()` for N-player scoring
2. **MCTS** (`agents/mcts.py`): Switch from scalar values to per-player value vectors
3. **GNN features** (`agents/carcassonne_gnn_agent.py`): Extend `MEEPLE_OWNER_DIM` to N+1, update game globals to include all players' resources
4. **Visualizer**: Add N player colors

### 12.3 Replacing the Search Algorithm

The system is modular around a single interface:

```python
agent.get_action_probs(state, temperature, add_exploration_noise) → Dict[action, float]
```

To replace MCTS with another search algorithm (e.g., alpha-beta), you need to modify only 3 files:

1. **`agents/mcts.py`** — Replace with your search class exposing `get_action_probs()`
2. **`agents/carcassonne_gnn_agent.py`** — Change `_prepare_mcts()` to instantiate your class
3. **`agents/mcts_ml_agent.py`** — Update delegation in `get_action_probs()` and `reset_inference()`

The neural network, training loop, game engine, arena, and checkpointing remain untouched. The search algorithm accesses the NN through an inference closure:

```python
def inference_fn(state, legal_actions) -> {'policy': Dict[action, float], 'value': float}
```

**Design consideration for alpha-beta**: MCTS naturally produces a policy distribution from visit counts. Alpha-beta produces a single best move. To generate training targets, you would need to run alpha-beta for each root child and apply softmax over minimax scores — losing alpha-beta's pruning advantage at the root level. See `README_STAR_MINIMAX.md` for the Star-Minimax extension that handles chance nodes (tile draws) in alpha-beta trees.

---

## Outputs and Checkpoints

All outputs are written to `experiments/runs/carcassonne/<timestamp>_<tag>/`:

```
checkpoints/
  <timestamp>_<iter>/
    best/                    # PyTorch model weights
    training_state.json      # {"current_iteration": N}
    training_data.json       # Serialized replay buffer
logs/                        # TensorBoard event files
```

**TensorBoard metrics** per iteration: self-play scores, win rates vs random, convergence diagnostics, learning rate.

---

## License

See [LICENSE](LICENSE).

# Training Algorithm: AlphaZero-Style Self-Play with MCTS

This document describes the training algorithm used in this project. It is an
adaptation of DeepMind's **AlphaZero** algorithm to the board game Carcassonne.
The neural network architecture is documented separately; this document focuses
exclusively on the search, self-play, and training loop.

---

## Table of Contents

1. [High-Level Overview](#1-high-level-overview)
2. [Monte Carlo Tree Search (MCTS)](#2-monte-carlo-tree-search-mcts)
   - 2.1 [Tree Structure](#21-tree-structure)
   - 2.2 [Selection (PUCT)](#22-selection-puct)
   - 2.3 [Expansion and Evaluation](#23-expansion-and-evaluation)
   - 2.4 [Backpropagation](#24-backpropagation)
   - 2.5 [Action Selection from the Root](#25-action-selection-from-the-root)
   - 2.6 [Exploration Noise](#26-exploration-noise)
   - 2.7 [Value Normalization](#27-value-normalization)
3. [Self-Play Episode](#3-self-play-episode)
4. [Training the Neural Network](#4-training-the-neural-network)
   - 4.1 [Training Targets](#41-training-targets)
   - 4.2 [Loss Function](#42-loss-function)
5. [Arena Evaluation and Model Gating](#5-arena-evaluation-and-model-gating)
6. [Outer Training Loop](#6-outer-training-loop)
7. [Differences from Classical Minimax](#7-differences-from-classical-minimax)
8. [Hyperparameters](#8-hyperparameters)
9. [Full Pseudocode](#9-full-pseudocode)
10. [Codebase Modularity and Replacing the Search Algorithm](#10-codebase-modularity-and-replacing-the-search-algorithm)
    - 10.1 [Architecture Layers and File Map](#101-architecture-layers-and-file-map)
    - 10.2 [The Critical Interface Contract](#102-the-critical-interface-contract)
    - 10.3 [How the Value Function Is Updated (End-to-End Data Flow)](#103-how-the-value-function-is-updated-end-to-end-data-flow)
    - 10.4 [How the Policy Is Updated (End-to-End Data Flow)](#104-how-the-policy-is-updated-end-to-end-data-flow)
    - 10.5 [What Stays Completely Unchanged](#105-what-stays-completely-unchanged)
    - 10.6 [What Must Change (and How Much Work)](#106-what-must-change-and-how-much-work)
    - 10.7 [Design Challenges for Alpha-Beta as a Replacement](#107-design-challenges-for-alpha-beta-as-a-replacement)
    - 10.8 [Step-by-Step Replacement Guide](#108-step-by-step-replacement-guide)

---

## 1. High-Level Overview

The algorithm learns to play Carcassonne entirely from self-play, with no human
game data. It consists of three interleaved components:

```
┌─────────────────────────────────────────────────────────┐
│                    OUTER LOOP (iterations)               │
│                                                         │
│  ┌───────────────┐    ┌───────────┐    ┌─────────────┐  │
│  │   Self-Play    │───▶│  Training  │───▶│    Arena     │  │
│  │  (MCTS + net)  │    │  (SGD on   │    │ (new model   │  │
│  │  generates     │    │  self-play │    │  vs. prev    │  │
│  │  training data │    │  data)     │    │  model)      │  │
│  └───────────────┘    └───────────┘    └─────────────┘  │
│         ▲                                     │          │
│         └─────────────────────────────────────┘          │
│                    accept / reject model                  │
└─────────────────────────────────────────────────────────┘
```

1. **Self-play**: The current neural network guides MCTS to play complete games
   against itself. Each turn produces a training example: the game state, the
   MCTS visit-count policy, and (after the game ends) the game outcome.

2. **Training**: The neural network is trained on the accumulated self-play data
   to better predict both the MCTS policy and the game outcome.

3. **Arena evaluation** (optional): The newly trained model plays head-to-head
   against the previous model. If it does not win often enough, the update is
   rejected and the previous model is restored.

---

## 2. Monte Carlo Tree Search (MCTS)

MCTS is the core decision-making algorithm. It builds a search tree
incrementally by running many simulations from the current game state, using
the neural network to evaluate leaf nodes. It does **not** enumerate the full
game tree (which is intractable for Carcassonne); instead it focuses
computational effort on the most promising branches.

### 2.1 Tree Structure

Each node in the search tree stores:

| Field            | Description                                                |
|------------------|------------------------------------------------------------|
| `state`          | The game state at this node                                |
| `visit_count`    | Number of times this node has been visited (N)             |
| `value_estimate` | Running mean of backpropagated values (Q)                  |
| `prior`          | The neural network's prior probability for this action (P) |
| `children`       | Map from legal actions to child nodes                      |
| `expanded`       | Whether this node has been evaluated by the network        |

The tree is rooted at the current game state. Children represent successor
states reached by taking specific actions.

```
              Root (current state, N=25)
             /          |            \
        a1 (N=12)    a2 (N=8)    a3 (N=5)
       /    \           |
   a11(N=7) a12(N=5)  a21(N=8)
    ...       ...       ...
```

### 2.2 Selection (PUCT)

Starting from the root, the algorithm descends through the tree by selecting
the child that maximizes the **PUCT** (Predictor + Upper Confidence bound for
Trees) score. This is the key formula that balances exploitation of
high-value moves against exploration of under-visited moves:

```
PUCT(parent, child) = Q(child) + c_puct * P(child) * sqrt(N(parent) - 1) / (1 + N(child))
```

Where:

- **Q(child)**: The child's current value estimate (running average of
  backpropagated values). For an unexpanded child, Q = 0.
- **P(child)**: The neural network's prior probability for the action leading
  to this child. Higher priors mean the network considers this move promising.
- **N(parent)**: The parent's visit count (minus 1, since we increment before
  selecting).
- **N(child)**: The child's visit count.
- **c_puct**: Exploration constant (default 1.25). Controls the
  exploitation-exploration trade-off.

**Intuition**: The first term (Q) favors moves that have historically led to
good outcomes. The second term favors moves with high prior probability that
have been visited relatively few times. As N(child) grows, the exploration
bonus shrinks, and the algorithm increasingly trusts Q.

**Two-player value negation**: Since the game alternates between players, a
child's value must be negated when the parent and child belong to different
players. A position that is good for the opponent is bad for the current
player:

```
if parent.current_player != child.current_player:
    Q(child) = -Q(child)
```

### 2.3 Expansion and Evaluation

When selection reaches an **unexpanded leaf node** (a node not yet evaluated by
the neural network), the algorithm:

1. Obtains the game state by applying the parent's action to the parent's
   state.
2. **If the state is terminal**: Records the actual game outcome as the node's
   value (no network evaluation needed).
3. **If the state is non-terminal**: Feeds the state to the neural network,
   which returns:
   - A **policy** P(a) over all legal actions (used as priors for child nodes).
   - A **value** V (the network's estimate of how good this state is for the
     current player).
4. Creates child nodes for each legal action, storing the predicted prior.
5. Returns the value as the **sample return** for backpropagation.

This is the only point where the neural network is called during search. Each
of the `num_search` simulations makes at most one network call.

### 2.4 Backpropagation

After evaluating a leaf, the sample return is propagated back up the path from
the leaf to the root. At each node along the path:

```
if visit_count == 1:            # first visit
    Q = sample_return
else:
    Q = (old_count * Q + sample_return) / visit_count
```

This maintains Q as the running mean of all sample returns that have passed
through this node.

**Player alternation**: When backpropagating from a child to a parent that
belongs to a different player, the sample return is negated:

```
if child.current_player != parent.current_player:
    sample_return = -sample_return
```

This ensures each node's Q reflects the value from the perspective of the
player at that node.

### 2.5 Action Selection from the Root

After all `num_search` simulations complete, the root's children have
accumulated visit counts. The final action is selected based on these counts:

**With temperature (tau > 0)** — used during self-play for exploration:

```
pi(a) = N(a)^(1/tau) / sum_b N(b)^(1/tau)
```

The action is then sampled from this distribution. Higher temperature means
more random exploration; tau = 1 is proportional to visit counts.

**Without temperature (tau = 0)** — used during arena evaluation for strength:

The action with the highest visit count is selected deterministically (ties
broken randomly).

### 2.6 Exploration Noise

To ensure the self-play agent explores diverse openings, **Dirichlet noise** is
added to the prior probabilities at the root node:

```
P'(a) = (1 - epsilon) * P(a) + epsilon * eta(a)
```

Where:

- `epsilon` = 0.05 (root exploration fraction).
- `eta` ~ Dir(alpha, alpha, ..., alpha) with `alpha` = 0.25.

This is only applied at the root, and only during self-play (not during arena
evaluation). It prevents the search from being completely determined by the
network's current prior, allowing it to discover moves the network might
undervalue.

### 2.7 Value Normalization

When enabled, MCTS normalizes the Q-values in the PUCT formula to `[0, 1]`
using the minimum and maximum values observed so far in the current search
tree:

```
Q_normalized = (Q - Q_min) / (Q_max - Q_min)
```

If Q_max = Q_min (all values are the same), Q_normalized = 0.

This is particularly important when using continuous value targets (such as
tanh-normalized score margins) rather than discrete {-1, 0, +1} outcomes,
because it prevents the magnitude of values from distorting the
exploitation-exploration balance in PUCT.

---

## 3. Self-Play Episode

A single self-play episode produces one complete game worth of training data.
The same neural network plays both sides.

**Step-by-step**:

1. Initialize the game to its starting state.
2. Set turn counter `i = 0`.
3. At each turn:
   - Increment `i`.
   - Set `temperature = 1` if `i < temperature_threshold`, else `temperature = 0`.
     (Early moves are exploratory; later moves are greedy.)
   - Run MCTS from the current state with `num_search` simulations and
     Dirichlet exploration noise.
   - Record the pair `(state, pi)` where `pi` is the MCTS visit-count policy.
   - Sample an action from `pi` and advance the game state.
4. When the game terminates, compute the game outcome `z` for the terminal
   player using the margin-normalized value:
   ```
   z = tanh((my_score - opponent_score) / margin_scale)
   ```
5. **Assign values to all recorded positions**: For each recorded
   `(state, pi)`, the training value is `z` if that state's current player is
   the same as the terminal state's current player, and `-z` otherwise.
6. Return the list of training examples `[(state, pi, v), ...]`.

---

## 4. Training the Neural Network

### 4.1 Training Targets

Each training example is a triple `(s, pi, v)`:

- **s**: A game state (converted to a graph for the GNN).
- **pi**: The MCTS visit-count policy — a probability distribution over legal
  actions. This is the "expert" policy that the network's policy head should
  learn to imitate.
- **v**: The game outcome from the perspective of the player at state s.
  This is the regression target for the network's value head.

The key insight: MCTS produces **better move probabilities** than the raw
network, because it looks ahead via tree search. By training the network to
match MCTS's output, the network improves. In the next iteration, a stronger
network produces better MCTS searches, which produce better training data,
creating a virtuous cycle.

### 4.2 Loss Function

The network has two heads: a **policy head** and a **value head**. The loss is
the sum of two terms:

```
L = L_policy + L_value
```

**Policy loss** (cross-entropy):

```
L_policy = -(1/B) * sum_i sum_a pi_i(a) * log(p_i(a))
```

Where `p_i(a)` is the network's predicted probability for action `a` in
sample `i`, and `pi_i(a)` is the MCTS target probability. Since the action
space varies per state, a grouped softmax is used: action scores are computed
for each sample's legal actions, then softmax is applied within each sample's
action group.

**Value loss** (mean squared error):

```
L_value = (1/B) * sum_i (v_i - v_hat_i)^2
```

Where `v_hat_i` is the network's predicted value and `v_i` is the game outcome
target.

The optimizer is Adam with learning rate 1e-3. Training runs for 20 epochs over
the accumulated self-play data each iteration.

---

## 5. Arena Evaluation and Model Gating

After training, the updated model is optionally tested against the previous
model in head-to-head matches (the "arena"):

1. Save the current model as a checkpoint.
2. Load the checkpoint into a separate "previous agent."
3. Play `arena_iterations` games between the new and previous agents, evenly
   split between who goes first (to eliminate first-player bias):
   - Half the games: new model = Player 0, previous model = Player 1.
   - Other half: swapped.
   - Arena games use `temperature = 0` (greedy play) and no exploration noise.
4. If the new model's win rate exceeds `arena_win_threshold`, accept the update.
   Otherwise, **reject** the update and restore the previous model's weights.

This gating mechanism prevents catastrophic forgetting: if a training update
accidentally makes the model weaker, it is rolled back.

After the model acceptance/rejection, the (potentially restored) model also
plays against a purely random agent to track absolute strength.

---

## 6. Outer Training Loop

The complete training procedure:

```
for iteration = 1, 2, ..., num_iters:

    1. SELF-PLAY
       Play num_self_play_episodes complete games using MCTS + current network.
       Accumulate training examples [(state, mcts_policy, outcome), ...].

    2. DATA MANAGEMENT
       Append new examples to a rolling buffer.
       Keep only the most recent max_train_data_iters iterations of data
       (older data is discarded to prevent learning from stale examples).

    3. TRAINING
       Train the network for train_epochs passes over the accumulated data
       using mini-batch SGD (Adam optimizer).

    4. ARENA EVALUATION (if arena_win_threshold > 0)
       Pit the trained network against its pre-training snapshot.
       Accept or reject the update based on win rate.

    5. CHECKPOINT
       Save the model, optimizer state, and training data to disk.

    6. RANDOM BASELINE
       Play the current model against a random agent and log win rate.
```

---

## 7. Differences from Classical Minimax

This algorithm is often mistaken for minimax with alpha-beta pruning, but they
are fundamentally different:

| Property | Minimax + Alpha-Beta | AlphaZero MCTS |
|----------|---------------------|----------------|
| **Search strategy** | Exhaustive depth-limited search of all branches | Selective sampling — focuses on promising branches |
| **Pruning** | Alpha-beta pruning eliminates branches that provably cannot affect the result | No pruning; under-visited branches are naturally deprioritized by PUCT |
| **Evaluation** | Handcrafted heuristic at leaf nodes | Neural network evaluates leaf nodes |
| **Branching** | Must consider all moves (up to pruning) | Focuses on high-prior, high-value moves via PUCT selection |
| **Learning** | No learning — evaluation function is fixed | Self-improving: network learns from its own MCTS-guided play |
| **Move ordering** | Critical for alpha-beta efficiency; often uses heuristics | Implicit via neural network priors — good moves are explored first |
| **Asymmetry** | Explores good branches deeply, prunes bad ones early | Explores good branches more frequently, but all branches remain reachable |
| **Stochasticity** | Deterministic (given fixed evaluation) | Stochastic: Dirichlet noise, temperature-based action sampling |

**Why MCTS over Minimax for Carcassonne?**

Carcassonne has an extremely high branching factor (hundreds of legal actions
per turn when considering tile placement, rotation, and meeple placement). It
also has stochastic elements (the next tile is drawn randomly from the deck).
Minimax with alpha-beta pruning is poorly suited to this setting because:

- The branching factor makes deep exhaustive search infeasible.
- Alpha-beta pruning is less effective with high branching factors.
- Stochastic elements (tile draws) require expectimax variants, further
  increasing the search space.

MCTS handles all of these naturally: it samples the most promising actions,
uses the neural network to evaluate positions without deep lookahead, and can
incorporate stochastic events (though in this implementation, MCTS searches
within a single turn where the drawn tile is known).

---

## 8. Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_search` | 25 | MCTS simulations per move |
| `cpuct` | 1.25 | Exploration constant in PUCT formula |
| `root_dirichlet_alpha` | 0.25 | Dirichlet noise concentration parameter |
| `root_exploration_fraction` | 0.05 | Weight of Dirichlet noise vs. network prior |
| `enable_value_normalization` | True | Normalize Q-values to [0,1] in PUCT |
| `temperature_threshold` | 60 | Turns before switching from exploratory to greedy play |
| `margin_scale` | 40.0 | Denominator in `tanh(score_diff / margin_scale)` for value targets |
| `num_iters` | 5 | Number of outer training iterations |
| `num_self_play_episodes` | 10 | Self-play games per iteration |
| `max_train_data_iters` | 4 | Rolling window of iterations to keep for training |
| `train_epochs` | 20 | SGD passes per training phase |
| `train_batch_size` | 128 | Mini-batch size |
| `learning_rate` | 1e-3 | Adam learning rate |
| `arena_iterations` | 20 | Head-to-head games for model gating |
| `arena_win_threshold` | 0 | Minimum win rate to accept new model (0 = disabled) |

---

## 9. Full Pseudocode

### 9.1 Outer Training Loop

```
function ALPHAZERO_TRAIN(num_iters, num_episodes, ...):
    network ← initialize randomly
    replay_buffer ← empty rolling buffer

    for iteration = 1 to num_iters:

        // ── Phase 1: Self-Play ──
        new_examples ← []
        for episode = 1 to num_episodes:
            examples ← SELF_PLAY_EPISODE(network)
            new_examples.append(examples)

        replay_buffer.append(new_examples)
        replay_buffer.discard_oldest_if_exceeds(max_iters)

        // ── Phase 2: Training ──
        snapshot ← copy(network)
        for epoch = 1 to train_epochs:
            for batch in shuffle_and_batch(replay_buffer):
                states, target_policies, target_values ← batch
                pred_policies, pred_values ← network(states)
                loss ← cross_entropy(pred_policies, target_policies)
                        + mse(pred_values, target_values)
                network.update(∇loss)

        // ── Phase 3: Arena Gating (optional) ──
        if arena_threshold > 0:
            wins ← ARENA(network, snapshot, num_arena_games)
            if wins / num_arena_games < arena_threshold:
                network ← snapshot    // reject update

        save_checkpoint(network)
```

### 9.2 Self-Play Episode

```
function SELF_PLAY_EPISODE(network):
    state ← game.initial_state()
    history ← []
    turn ← 0

    while not state.is_terminal():
        turn ← turn + 1
        tau ← 1 if turn < temperature_threshold else 0

        pi ← MCTS_SEARCH(state, network, tau)
        history.append((state, pi))

        action ← sample(pi)
        state ← state.apply(action)

    // Assign outcome to all positions
    z_terminal ← tanh(score_diff / margin_scale)
    terminal_player ← state.current_player

    examples ← []
    for (s, pi) in history:
        if s.current_player == terminal_player:
            z ← z_terminal
        else:
            z ← -z_terminal
        examples.append((s, pi, z))

    return examples
```

### 9.3 MCTS Search

```
function MCTS_SEARCH(state, network, temperature):
    root ← new Node(state)

    // Add Dirichlet exploration noise to root priors
    noise ← Dirichlet(alpha, ..., alpha)   // one per legal action
    root.prior[a] ← (1 - eps) * root.prior[a] + eps * noise[a]  for each a

    Q_min ← +∞
    Q_max ← -∞

    for simulation = 1 to num_search:
        SIMULATE(root)

    // Convert visit counts to policy
    counts ← {a: root.children[a].visit_count for a in legal_actions}
    if temperature == 0:
        return one_hot(argmax(counts))       // greedy
    else:
        pi[a] ← counts[a]^(1/tau) / Σ_b counts[b]^(1/tau)
        return pi
```

### 9.4 Single MCTS Simulation (Recursive)

```
function SIMULATE(node) → sample_return:
    node.visit_count += 1

    if node.is_expanded:
        if node.state.is_terminal():
            return node.value_estimate       // actual game outcome

        // ── Selection ──
        best_child ← argmax over children c of:
            PUCT_SCORE(node, c)

        child_return ← SIMULATE(best_child)

        // Negate if players differ
        if node.current_player ≠ best_child.current_player:
            child_return ← -child_return

        // ── Backpropagation ──
        old_count ← node.visit_count - 1
        if old_count == 0:
            node.Q ← child_return
        else:
            node.Q ← (old_count * node.Q + child_return) / node.visit_count

        UPDATE_MIN_MAX(node)
        return child_return

    else:
        // ── Expansion & Evaluation ──
        state ← node.parent.state.apply(node.action)

        if state.is_terminal():
            value ← state.get_player_value(state.current_player)
            node.expand(state, policy={}, value)
            sample_return ← value

        else:
            policy, value ← network(state)
            node.expand(state, policy, value)
            sample_return ← value

        UPDATE_MIN_MAX(node)
        return sample_return
```

### 9.5 PUCT Selection Score

```
function PUCT_SCORE(parent, child):
    if child.is_expanded:
        Q ← child.Q
        // Negate for opponent
        if parent.current_player ≠ child.current_player:
            Q ← -Q
        // Normalize to [0, 1]
        if value_normalization_enabled and Q_max > Q_min:
            Q ← (Q - Q_min) / (Q_max - Q_min)
        else if value_normalization_enabled:
            Q ← 0
    else:
        Q ← 0

    P ← child.prior
    // Apply Dirichlet noise at root
    if parent == root and exploration_noise_enabled:
        P ← (1 - eps) * P + eps * dirichlet_noise[child.action]

    exploration ← c_puct * P * sqrt(parent.visit_count - 1) / (1 + child.visit_count)

    return Q + exploration
```

### 9.6 Arena Evaluation

```
function ARENA(new_network, old_network, num_games):
    new_wins ← 0
    old_wins ← 0
    draws ← 0

    // Play half the games with new as Player 0
    for game = 1 to num_games / 2:
        winner ← PLAY_GAME(players=[new_network, old_network], temperature=0)
        if winner == 0: new_wins += 1
        elif winner == 1: old_wins += 1
        else: draws += 1

    // Play other half with new as Player 1
    for game = 1 to num_games / 2:
        winner ← PLAY_GAME(players=[old_network, new_network], temperature=0)
        if winner == 1: new_wins += 1
        elif winner == 0: old_wins += 1
        else: draws += 1

    return new_wins, old_wins, draws
```

---

## 10. Codebase Modularity and Replacing the Search Algorithm

This section describes the exact boundaries between the search algorithm (MCTS)
and the rest of the system, to answer the question: **"If I want to replace
MCTS with something else (e.g., alpha-beta pruning), what do I touch and what
stays the same?"**

### 10.1 Architecture Layers and File Map

The codebase is organized into five layers. MCTS lives in only one of them.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ Layer 5: Training Loop                                                      │
│   carcassonne_train.py                                                      │
│   - Runs self-play episodes, collects (state, policy, value) tuples         │
│   - Calls agent.get_action_probs() — does NOT know what algorithm is inside │
│   - Feeds training data to agent.train()                                    │
│   - Runs arena games                                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│ Layer 4: Carcassonne-Specific Agent                                         │
│   agents/carcassonne_gnn_agent.py                                           │
│   - Neural network model definitions (GPSCarcassonneModel, etc.)            │
│   - build_graph_from_state() — converts GameState to PyG Data               │
│   - _prepare_mcts() — creates the MCTS with an inference closure            │
│   - _train_step_gnn() — batched training with variable action spaces        │
│   - train() — epoch loop calling _train_step_gnn()                          │
├─────────────────────────────────────────────────────────────────────────────┤
│ Layer 3: Agent Wiring (base class)                                          │
│   agents/mcts_ml_agent.py                                                   │
│   - MCTSMLAgent base class                                                  │
│   - get_action_probs() delegates to self.mcts.get_action_probs()            │
│   - _prepare_mcts() — overridden by subclasses                              │
│   - Owns the optimizer and model                                            │
├─────────────────────────────────────────────────────────────────────────────┤
│ Layer 2: Search Algorithm                      ◄── THIS IS WHAT YOU REPLACE │
│   agents/mcts.py                                                            │
│   - MCTSConfig dataclass                                                    │
│   - Node class (tree structure)                                             │
│   - MCTS class (selection, expansion, backpropagation, action selection)    │
│   - get_action_probs(state, temperature, noise) → Dict[action, float]       │
├─────────────────────────────────────────────────────────────────────────────┤
│ Layer 1: Game Engine                                                        │
│   games/game_state.py — abstract GameState interface                        │
│   games/carcassonne_game_state.py — Carcassonne wrapper                     │
│   carcassonne/ — full game engine (tiles, rules, scoring, etc.)             │
│   - get_legal_actions(), apply_action(), is_terminal()                      │
│   - get_player_value(), get_current_player()                                │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 10.2 The Critical Interface Contract

The entire system is glued together by a single interface. Everything above
Layer 2 only sees:

```python
agent.get_action_probs(state, temperature, add_exploration_noise) → Dict[action, float]
```

This returns a probability distribution over legal actions. The training loop
(`carcassonne_train.py`) calls this function, samples an action from the
returned distribution, records the distribution as a training target, and moves
on. **It does not know or care whether MCTS, alpha-beta, or a random number
generator produced that distribution.**

Internally, `get_action_probs` works as follows (currently):

```
carcassonne_train.py
    └── agent.get_action_probs(state, temperature, noise)     [Layer 5 → Layer 3]
            └── self.mcts.get_action_probs(state, temp, noise) [Layer 3 → Layer 2]
                    └── for i in range(num_search):
                            self._search(root)                 [Layer 2 internal]
                                └── self.model(state, actions)  [Layer 2 → inference_fn]
                                        └── network(data, actions) [inference_fn → Layer 4 model]
```

The search algorithm (Layer 2) interacts with the neural network through an
**inference function** (a closure), not through the model directly. The
inference function has this signature:

```python
def inference_fn(state: GameState, all_legal_actions: List[Any]) -> dict:
    # Returns {'policy': Dict[action, float], 'value': float}
```

Where `policy` is a normalized probability distribution over `all_legal_actions`
and `value` is a scalar estimate of the current player's advantage.

### 10.3 How the Value Function Is Updated (End-to-End Data Flow)

The value head learns to predict game outcomes. Here is the complete data flow:

```
                            ┌──────────────────────────────────────┐
                            │       SELF-PLAY GAME FINISHES        │
                            │  Player 0 scores 80, Player 1: 50    │
                            └──────────┬───────────────────────────┘
                                       │
                                       ▼
               ┌──────────────────────────────────────────────┐
               │  get_player_value(player)                     │
               │  diff = scores[player] - scores[opponent]     │
               │  return tanh(diff / 40.0)                     │
               │                                               │
               │  Player 0 value: tanh(30/40) ≈ +0.64          │
               │  Player 1 value: tanh(-30/40) ≈ -0.64         │
               │                                               │
               │  File: games/carcassonne_game_state.py:31     │
               └──────────┬───────────────────────────────────┘
                          │
                          ▼
           ┌──────────────────────────────────────────────────┐
           │  PROPAGATE TO ALL STATES IN EPISODE               │
           │                                                   │
           │  For each recorded (state, pi):                   │
           │    if state.current_player == terminal_player:    │
           │        v = z_terminal                             │
           │    else:                                          │
           │        v = -z_terminal                            │
           │                                                   │
           │  File: carcassonne_train.py:131-132               │
           └──────────┬───────────────────────────────────────┘
                      │
                      ▼
         ┌──────────────────────────────────────────────────────┐
         │  TRAINING: build value targets tensor                 │
         │                                                       │
         │  value_targets = tensor([v₁, v₂, ..., vB])           │
         │  pred_values = model.predict_value(node_embeddings)   │
         │  value_loss = MSE(pred_values, value_targets)         │
         │                                                       │
         │  File: agents/carcassonne_gnn_agent.py:1262-1265     │
         └──────────┬───────────────────────────────────────────┘
                    │
                    ▼
         ┌──────────────────────────────────────────────────────┐
         │  GRADIENT UPDATE                                      │
         │                                                       │
         │  loss = policy_loss + value_loss                      │
         │  loss.backward()                                      │
         │  optimizer.step()                                     │
         │                                                       │
         │  Gradient flows through: value_loss → predict_value   │
         │  → encode (GPS layers) → node_proj, rwpe_enc, etc.   │
         │                                                       │
         │  The value head AND the shared encoder both update.   │
         └──────────────────────────────────────────────────────┘
```

**Key point for replacement**: The value training targets come from the
**actual game outcome**, not from the search algorithm's value estimate. MCTS
does not directly produce the value training target. It only influences the
value indirectly by determining which actions are played (which affects the
final score). This means **the value training pipeline is completely
search-algorithm-agnostic**. Whether you use MCTS, alpha-beta, or random moves
to play the self-play games, the value targets are always computed the same way
from the final score.

During inference, the search algorithm does use the value head's predictions:
MCTS calls `inference_fn → network → pred_value` to evaluate leaf nodes. An
alpha-beta replacement would use the value head in the same way — as the
leaf evaluation function.

### 10.4 How the Policy Is Updated (End-to-End Data Flow)

The policy head learns to predict the search algorithm's move preferences. This
is where the search algorithm has a **direct** impact on training:

```
                     ┌───────────────────────────────────────┐
                     │          MCTS SEARCH (25 sims)        │
                     │                                       │
                     │  Root has 3 legal actions:             │
                     │    a1: visited 15 times                │
                     │    a2: visited 8 times                 │
                     │    a3: visited 2 times                 │
                     │                                       │
                     │  File: agents/mcts.py:235-241         │
                     └──────────┬────────────────────────────┘
                                │
                                ▼
                  ┌───────────────────────────────────────────┐
                  │  CONVERT VISIT COUNTS → POLICY (pi)       │
                  │                                           │
                  │  With temperature=1:                      │
                  │    pi = {a1: 15/25, a2: 8/25, a3: 2/25}  │
                  │        = {a1: 0.60, a2: 0.32, a3: 0.08}  │
                  │                                           │
                  │  File: agents/mcts.py:253-258             │
                  └──────────┬────────────────────────────────┘
                             │
                             ▼
              ┌───────────────────────────────────────────────┐
              │  STORE AS TRAINING TARGET                      │
              │                                                │
              │  train_data.append((state, pi))                │
              │                                                │
              │  File: carcassonne_train.py:123                │
              └──────────┬────────────────────────────────────┘
                         │
                         ▼
         ┌───────────────────────────────────────────────────────┐
         │  TRAINING: score each legal action, apply softmax     │
         │                                                       │
         │  For each sample i in the batch:                      │
         │    scores_i = model.score_actions(embeddings, actions) │
         │    // e.g., scores = [2.1, 1.3, -0.5]                │
         │                                                       │
         │  All scores concatenated, softmax within each sample: │
         │    probs = grouped_softmax(all_scores, batch_indices) │
         │    // e.g., probs = [0.55, 0.33, 0.12]               │
         │                                                       │
         │  Cross-entropy against MCTS targets:                  │
         │    policy_loss = -Σ pi(a) * log(probs(a)) / B        │
         │    // = -(0.60*log(0.55) + 0.32*log(0.33)            │
         │    //    + 0.08*log(0.12)) / B                        │
         │                                                       │
         │  File: agents/carcassonne_gnn_agent.py:1241-1260     │
         └──────────┬───────────────────────────────────────────┘
                    │
                    ▼
         ┌───────────────────────────────────────────────────────┐
         │  GRADIENT UPDATE                                      │
         │                                                       │
         │  loss = policy_loss + value_loss                      │
         │  loss.backward()                                      │
         │  optimizer.step()                                     │
         │                                                       │
         │  Gradient flows through: policy_loss → score_actions  │
         │  (tile_scorer / meeple_scorer) → encode (GPS layers)  │
         │                                                       │
         │  The policy scoring heads AND the shared encoder      │
         │  both update.                                         │
         └───────────────────────────────────────────────────────┘
```

**Key point for replacement**: The policy target `pi` is the **only** thing
that comes directly from the search algorithm. MCTS produces it from visit
counts. A replacement algorithm must also produce a `Dict[action, float]`
probability distribution to serve as the policy training target. This is the
central design challenge (discussed in Section 10.7).

**The virtuous cycle**: The network's policy head produces priors `P(a)` that
guide the search (via PUCT). Better priors lead to better search, which
produces better policy targets `pi`, which train better priors. This cycle is
the engine of AlphaZero's self-improvement. Any replacement algorithm must
preserve this cycle or replace it with an equivalent feedback mechanism.

### 10.5 What Stays Completely Unchanged

| Component | File(s) | Why it's unaffected |
|-----------|---------|---------------------|
| Game engine | `carcassonne/` (entire package) | Search algorithms consume `GameState`; the engine doesn't know about search |
| Game state wrapper | `games/carcassonne_game_state.py` | Provides `get_legal_actions()`, `apply_action()`, `is_terminal()`, `get_player_value()` — pure game logic |
| GameState interface | `games/game_state.py` | Abstract interface, no search logic |
| Neural network architecture | `agents/carcassonne_gnn_agent.py` (model classes: `GPSCarcassonneModel`, `GNNCarcassonneModel`) | The network's `encode()`, `predict_value()`, `score_actions()` methods are called by the training code, not by MCTS |
| Graph construction | `agents/carcassonne_gnn_agent.py` (`build_graph_from_state()`) | Converts `GameState` → PyG `Data` object; search-independent |
| Training loss computation | `agents/carcassonne_gnn_agent.py` (`_train_step_gnn()`) | Consumes `(state, action_probs_dict, value_float)` tuples; doesn't care how the `action_probs_dict` was produced |
| Training epoch loop | `agents/carcassonne_gnn_agent.py` (`train()`) | Loops over data, calls `_train_step_gnn()` |
| Outer training loop | `carcassonne_train.py` (lines 255-303) | Orchestrates self-play → train → arena; only calls `agent.get_action_probs()` |
| Arena game logic | `carcassonne_train.py` (`play_arena_games()`) | Only calls `agent.get_action_probs(state, temperature=0)` |
| Self-play data collection | `carcassonne_train.py` (`execute_episode()`) | Calls `agent.get_action_probs()`, records the result; search-agnostic |
| Value target computation | `carcassonne_train.py:131-132` | Computed from final game score, not from search |
| Checkpointing | `utils/utils.py`, `carcassonne_train.py` (`save()`, `load()`) | Saves/loads model weights; search-agnostic |
| Visualizer, test suites | `game_visualizer.py`, `checking_game_engine.py` | No interaction with search |
| Agent base classes | `agents/agent.py`, `agents/ml_agent.py` | Define `get_action_probs()` and `train()` abstract methods |

### 10.6 What Must Change (and How Much Work)

There are exactly **3 files** you need to modify, with varying degrees of effort:

#### File 1: `agents/mcts.py` — REPLACE (heavy, ~260 lines)

This is the core of the change. You would either:

- **Replace the entire file** with a new `alpha_beta.py` containing your search
  algorithm, or
- **Rewrite the internals** of the `MCTS` class.

Your replacement must expose a class with this method:

```python
class YourSearch:
    def __init__(self, model_fn, config):
        """
        model_fn: callable with signature
            (state: GameState, legal_actions: List) → {'policy': Dict, 'value': float}
        """
        ...

    def get_action_probs(self, state: GameState, temperature: float,
                         add_exploration_noise: bool) -> Dict[Any, float]:
        """
        Must return a normalized probability distribution over legal actions.
        The sum of values must equal 1.0.
        """
        ...
```

This is the only hard-contract requirement. The `model_fn` gives you access to
the neural network's policy and value predictions for any state.

#### File 2: `agents/carcassonne_gnn_agent.py` — MODIFY (light, ~10 lines)

Only the `_prepare_mcts()` method (lines 1167-1178) needs to change. It
currently creates an `MCTS` instance; you'd create your replacement instead:

```python
# Current (MCTS):
def _prepare_mcts(self) -> MCTS:
    def inference_fn(state, all_legal_actions):
        data = build_graph_from_state(state, use_gps=self.use_gps).to(self.device)
        self.model.eval()
        with torch.no_grad():
            with self.model_summarizer:
                result = self.model(data, all_legal_actions)
        return {'policy': result['policy'], 'value': result['value']}
    return MCTS(inference_fn, self.config.mcts_config)

# Replacement (alpha-beta):
def _prepare_search(self) -> YourSearch:
    def inference_fn(state, all_legal_actions):
        # Same inference function — this doesn't change
        data = build_graph_from_state(state, use_gps=self.use_gps).to(self.device)
        self.model.eval()
        with torch.no_grad():
            with self.model_summarizer:
                result = self.model(data, all_legal_actions)
        return {'policy': result['policy'], 'value': result['value']}
    return YourSearch(inference_fn, your_config)
```

The inference function itself (how the neural network is called) stays exactly
the same.

#### File 3: `agents/mcts_ml_agent.py` — MODIFY (light, ~5 lines)

Update the `get_action_probs()` delegation and `reset_inference()`:

```python
# Current:
def reset_inference(self):
    self.mcts = self._prepare_mcts()

def get_action_probs(self, state, temperature, add_exploration_noise):
    return self.mcts.get_action_probs(state, temperature, add_exploration_noise)

# Change to:
def reset_inference(self):
    self.search = self._prepare_search()

def get_action_probs(self, state, temperature, add_exploration_noise):
    return self.search.get_action_probs(state, temperature, add_exploration_noise)
```

The `MCTSMLAgentConfig` dataclass would also need its `mcts_config` field
replaced with your search config.

**Nothing else changes.** The training loop, loss function, neural network,
game engine, arena, checkpointing — all untouched.

### 10.7 Design Challenges for Alpha-Beta as a Replacement

Alpha-beta pruning is designed for a fundamentally different purpose than MCTS,
and fitting it into this training framework raises several non-trivial design
questions:

#### Challenge 1: Producing a Policy Distribution

MCTS naturally produces a probability distribution from visit counts:

```
pi(a) = N(a)^(1/tau) / Σ N(b)^(1/tau)
```

Alpha-beta produces a **single best move** and a minimax value. It does not
explore all children equally — it prunes branches that cannot affect the
result. To produce a training target for the policy head, you have several
options:

| Strategy | How it works | Pros | Cons |
|----------|-------------|------|------|
| **One-hot best move** | `pi = {best_action: 1.0}` | Simple | Very sparse signal; the network doesn't learn relative quality of non-best moves; cross-entropy gradient is zero for all other actions |
| **Softmax over alpha-beta scores** | Run alpha-beta for each root child to get values `v(a)`, then `pi(a) = exp(v(a)/tau) / Σ exp(v(b)/tau)` | Smooth distribution; teaches relative move quality | Requires running alpha-beta for EVERY root child (no pruning benefit at root level); very expensive |
| **Multi-PV search** | Run alpha-beta multiple times, each time excluding the previously found best move | Ranks top-K moves | K separate searches; still no smooth distribution for all moves |
| **Move-ordering scores** | Use the network's raw policy as the distribution and only use alpha-beta for the value | Cheapest | Policy head never improves beyond the raw network; no virtuous cycle |

**Recommendation**: The "softmax over alpha-beta scores" approach is the most
compatible with the existing training pipeline, because it produces a genuine
probability distribution that the cross-entropy loss can learn from. However,
it is expensive — you must evaluate each root child with a full alpha-beta
call, which defeats much of alpha-beta's pruning advantage at the root.

#### Challenge 2: Depth Limitation

Alpha-beta requires a fixed search depth (or iterative deepening). With
Carcassonne's branching factor of 50-200+ legal actions per turn, even a
depth-2 search expands 2,500-40,000 nodes per move. At depth 3, it's
125,000-8,000,000. Compare this to MCTS's fixed budget of 25 simulations
(25 neural network evaluations per move).

To make alpha-beta practical, you would likely need:

- Very shallow search (depth 1-2), relying heavily on the neural network's
  value estimate as the evaluation function.
- Aggressive move ordering using the neural network's policy prior (to maximize
  alpha-beta pruning).
- A time or node budget rather than a fixed depth.

#### Challenge 3: Stochastic Elements

Carcassonne has a hidden random element: the next tile is drawn from a shuffled
deck. Within a single turn the tile is known, but future turns are unknown.
Alpha-beta assumes a deterministic game. To handle Carcassonne's stochasticity,
you would need either:

- **Expectimax** (replace min/max at chance nodes with expected value over
  possible tile draws) — exponentially increases the search space.
- **Determinization** (assume a specific future tile order and run standard
  alpha-beta) — biased but practical.
- **Shallow search** (depth 1-2, avoiding the stochastic horizon entirely) —
  the simplest option.

MCTS handles this naturally because each simulation only expands one leaf;
it doesn't need to enumerate all possible tile draws.

#### Challenge 4: The Virtuous Cycle

In AlphaZero, the virtuous cycle works as follows:

1. Network priors `P(a)` guide MCTS toward good moves.
2. MCTS visit counts produce improved policy targets `pi(a)`.
3. Network is trained to predict `pi(a)`, improving `P(a)`.
4. Better `P(a)` → better MCTS → better `pi(a)` → ...

With alpha-beta, the cycle would be:

1. Network value `V(s)` serves as the leaf evaluation function.
2. Alpha-beta produces a minimax value and best move(s).
3. Network is trained to predict the alpha-beta policy and game outcome.
4. Better `V(s)` → deeper effective search → better moves → ...

This can work, but the policy head's role changes significantly. In MCTS, the
policy head is critical for search efficiency (it determines which branches
MCTS explores). In alpha-beta, the policy head is only used for move ordering
(to improve pruning efficiency). The value head becomes the primary driver of
search quality.

### 10.8 Step-by-Step Replacement Guide

If you decide to replace MCTS with a modified alpha-beta, here is the minimal
sequence of changes:

**Step 1**: Create `agents/alpha_beta.py` with a class that exposes
`get_action_probs(state, temperature, add_exploration_noise) → Dict[action, float]`.
The constructor should accept an `inference_fn` and a config object.

**Step 2**: In `agents/carcassonne_gnn_agent.py`, change `_prepare_mcts()` to
instantiate your new class instead of `MCTS`. The inference function closure
stays identical.

**Step 3**: In `agents/mcts_ml_agent.py`, update `reset_inference()` and
`get_action_probs()` to use your new class name. Update `MCTSMLAgentConfig` to
hold your config instead of `MCTSConfig`.

**Step 4**: Update the import in `agents/carcassonne_gnn_agent.py` from
`from .mcts import MCTS, MCTSConfig` to your new module.

**Step 5**: Test that `agent.get_action_probs(state, temperature=1)` returns a
valid probability distribution. The rest of the pipeline (training, arena,
checkpointing) will work automatically.

**Files you will NOT touch:**

- `carcassonne_train.py` — zero changes needed
- `agents/carcassonne_gnn_agent.py` model classes — zero changes needed
- `agents/carcassonne_gnn_agent.py` `_train_step_gnn()` — zero changes needed
- `games/` — zero changes needed
- `carcassonne/` — zero changes needed
- `utils/` — zero changes needed

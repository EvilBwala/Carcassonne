# The Star-Minimax Search Procedures

This document explains the Star1, Star2, and Star2.5 algorithms from Ballard
(1983), "The *-Minimax Search Procedure for Trees Containing Chance Nodes."
These algorithms extend the classical alpha-beta pruning strategy to game trees
that include **chance nodes** (e.g., dice rolls, card draws) — making them
relevant to games like Carcassonne where tiles are drawn randomly.

---

## Table of Contents

1. [The Problem: Chance in Game Trees](#1-the-problem-chance-in-game-trees)
2. [Background: Alpha-Beta Pruning](#2-background-alpha-beta-pruning)
3. [The Key Insight: Bounding Star-Node Values](#3-the-key-insight-bounding-star-node-values)
4. [Star1: Direct Extension of Alpha-Beta](#4-star1-direct-extension-of-alpha-beta)
5. [Star2: Probing for Earlier Cutoffs](#5-star2-probing-for-earlier-cutoffs)
6. [Star2.5: Iterative Probing](#6-star25-iterative-probing)
7. [Performance Comparison](#7-performance-comparison)
8. [Pseudocode](#8-pseudocode)

---

## 1. The Problem: Chance in Game Trees

In a standard minimax game tree, nodes alternate between two types:

- **+ nodes** (MAX): the maximizing player picks the child with the highest
  value.
- **- nodes** (MIN): the minimizing player picks the child with the lowest
  value.

Many games, however, include random events: dice rolls in backgammon, tile
draws in Carcassonne, card deals in blackjack. These are modeled by introducing
a third node type:

- **\* nodes** (CHANCE / STAR): the node's value is the **weighted average** of
  its children's values, where the weights are the probabilities of each
  outcome.

A tree containing all three node types is called a **\*-minimax tree**. For
example:

```
        + (MAX player picks best)
       / \
      *     *  (chance: average of children)
     / \   / \
    -   -  -   -  (MIN player picks worst)
   /\ /\ /\ /\
  leaves...        (terminal evaluations)
```

The value of a \* node with N equally likely children having values
V1, V2, ..., VN is:

```
value(*) = (V1 + V2 + ... + VN) / N
```

**The search problem**: Standard alpha-beta pruning cannot cut off branches
below \* nodes because the \* node's value depends on ALL its children (it's an
average, not a min or max). Naively, every child of every \* node must be
evaluated. The Star procedures solve this by deriving **bounds** on the \*
node's value after seeing only some of its children, enabling cutoffs.

---

## 2. Background: Alpha-Beta Pruning

To understand the Star procedures, recall how standard alpha-beta pruning
works:

- **Alpha (α)**: the best value that the MAX player can guarantee so far,
  along any path leading to this node. "MAX already has at least this good an
  option elsewhere."
- **Beta (β)**: the best value that the MIN player can guarantee so far.
  "MIN already has at least this good an option elsewhere."

A cutoff occurs when the value of a node is determined to be:
- **≥ β** at a MAX node: MIN will never allow play to reach here, because MIN
  already has a path that gives MAX less than β.
- **≤ α** at a MIN node: MAX will never allow play to reach here, because MAX
  already has a path worth at least α.

**Example:**

```
    + (MAX)
   / \
  4   ?          ← alpha = 4 after searching left child
      |
      - (MIN)
     / \
    3   ?        ← 3 ≤ alpha(4), so MIN node value is ≤ 3.
                   MAX already has 4, so this branch is irrelevant.
                   CUT OFF: don't search "?".
```

The problem: this reasoning breaks down at \* nodes. If a \* node has children
with values 3, ?, and ?, you can't conclude anything about the \* node's value
from just seeing the 3 — the unsearched children might be anything. Or can you?

---

## 3. The Key Insight: Bounding Star-Node Values

The central insight of the paper: even though a \* node's value is an average
of ALL its children, if you know the **range of possible leaf values** [L, U]
(the global minimum and maximum that any game evaluation can take), then after
seeing some children you can compute **lower and upper bounds** on the \* node.

Suppose a \* node has N equally likely children. You've evaluated the first
(i-1) children and obtained values V1, V2, ..., V(i-1). The remaining (N-i+1)
children haven't been searched yet. Their values must lie in [L, U]. Therefore:

```
Lower bound = (V1 + ... + V(i-1) + L * (N - i + 1)) / N
Upper bound = (V1 + ... + V(i-1) + U * (N - i + 1)) / N
```

The lower bound assumes every unsearched child has the minimum possible value;
the upper bound assumes every unsearched child has the maximum.

**Cutoff conditions:**

- If the **lower bound ≥ β**: the \* node's value is at least β, no matter
  what the unsearched children turn out to be. The MIN player above will never
  allow play to reach this node. **Cut off.**
- If the **upper bound ≤ α**: the \* node's value is at most α, no matter
  what. The MAX player above already has a better option. **Cut off.**

As more children are evaluated, the bounds tighten (L and U terms are replaced
by actual values), making cutoffs increasingly likely.

**Derived alpha-beta for children:** Before searching the ith child of a \*
node, you can also compute tighter alpha-beta bounds to pass down. If the ith
child's value V(i) must satisfy certain conditions for a cutoff NOT to occur,
those conditions become the new α and β for the subtree below the ith child.

---

## 4. Star1: Direct Extension of Alpha-Beta

**Star1** is the most straightforward algorithm. It processes the children of a
\* node left-to-right, maintaining running lower and upper bounds, and cuts off
when the bounds prove that the \* node's value is outside [α, β].

### How Star1 works:

Given a \* node with N children, alpha value α, beta value β, and leaf value
range [L, U]:

**Initialize:**

```
A = N * (α - U) + U       ← derived alpha for the 1st child
B = N * (β - L) + L       ← derived beta for the 1st child
vsum = 0                   ← running sum of evaluated children
```

**For each child i (left to right):**

1. Clamp: `AX = max(A, L)`, `BX = min(B, U)`.
2. Recursively evaluate child i with bounds [AX, BX], getting value v.
3. **Cutoff check:**
   - If `v ≤ A`: lower bound proves \* value ≤ α. Return α. (Cut off.)
   - If `v ≥ B`: upper bound proves \* value ≥ β. Return β. (Cut off.)
4. Accumulate: `vsum += v`.
5. **Tighten bounds for the next child:**
   - `A = A + U - v` (the unsearched-child upper bound shrinks by one term)
   - `B = B + L - v` (the unsearched-child lower bound shrinks by one term)

**If all children evaluated:** return `vsum / N`.

### Intuition for the bound update:

When you evaluate child i and get value v, the next child i+1 has one fewer
unsearched sibling to "rescue" the \* node's value. So the threshold for cutoff
shifts. If v was lower than U (the max possible), A increases — making a cutoff
easier. If v was higher than L, B decreases — again making a cutoff easier.

### Performance:

- **Best case** (optimal ordering): examines ~72.1% of leaves — a 28% savings.
- **Average case** (random ordering): examines ~79% of leaves — a 21% savings.
- Reduces to standard alpha-beta when there are no \* nodes.

---

## 5. Star2: Probing for Earlier Cutoffs

Star1 has a limitation: it evaluates each child of the \* node fully before
moving to the next. For a \* node whose children are all MIN nodes (a
"regular" \*-minimax tree where + and - alternate with \* interspersed), Star1
must search ALL N successors of each MIN node before it can move to the next
MIN node under the \* node. This is expensive.

**Star2's key idea**: before doing a full search of any MIN node, do a quick
**probe** — evaluate just ONE child of each MIN node. This gives a cheap upper
bound on each MIN node's value (since a MIN node's value is at most the value
of any single child). These tighter bounds make cutoffs happen much sooner.

### The two-phase approach:

**Phase 1 — Probing:** For each MIN node (child of the \* node), pick one
child (randomly or via heuristic) and evaluate it. Call this probe value w[i].
Since w[i] is a value of one child of a MIN node, the MIN node's true value is
at most w[i] (MIN will choose this child or something even lower).

After each probe, recompute the \* node's lower bound using these w[i] values
instead of U. Since w[i] ≤ U, the bound is tighter:

```
Star1 assumed:  unsearched MIN node value ≤ U
Star2 knows:    unsearched MIN node value ≤ w[i]
```

If the tighter lower bound exceeds β, cut off immediately — before doing any
full search.

**Phase 2 — Full search:** If probing didn't produce a cutoff, fall back to
Star1-style full search of each MIN node, but using the tighter A bound
derived from the probe values. Even when probing doesn't directly cause a
cutoff, the tighter bounds mean full-search cutoffs happen earlier.

### Why this works so well:

If the \* node being searched is **worse** than a previously searched \* node
(which is the common case — only the best \* node must be fully searched), the
probe values quickly reveal this. Instead of spending N evaluations per MIN
node to discover something is bad, Star2 spends just 1 evaluation per MIN
node.

### Performance:

- **Best case** (optimal ordering): O(N²) instead of O(N³) — an **order of
  magnitude** reduction.
- **Average case** (random ordering): examines ~35-53% of leaves (depending
  on branching factor) — **more than 50% reduction** compared to Star1 for
  branching factor > 20.
- With branching factor 40: Star1 examines 78.8%, Star2 examines 35.0%.

### Important restriction:

Star2 only applies to **regular** \*-minimax trees, where all children of a \*
node are the same type (all + or all -). This covers most games where chance
determines legal moves or outcomes but not who moves next — including
backgammon and Carcassonne.

---

## 6. Star2.5: Iterative Probing

Star2 does one probe per MIN node, then falls back to full search. Star2.5
extends this with **deeper probing** before committing to full search.

### Star2.5 Cyclic

Instead of probing one child then exhaustively searching, cycle through the MIN
nodes multiple times:

- **Round 1**: Look at 1 child of each MIN node (= Star2's probing phase).
- **Round 2**: If no cutoff, look at 1 MORE child of each MIN node.
- **Round 3**: If no cutoff, look at 1 MORE child of each MIN node.
- ...continue until either a cutoff occurs or all children have been examined.

The **probing factor** controls how many rounds of probing before falling back
to exhaustive search. With probing factor 1, this is Star2. With probing
factor 0, it degrades to Star1.

**Downside**: This requires breadth-first traversal of the MIN nodes (jumping
between MIN nodes between rounds), which incurs time and space overhead.

### Star2.5 Sequential

A simpler variant to avoid the breadth-first overhead:

- For each MIN node, probe **several** children (controlled by the probing
  factor) in one pass, then fall back to exhaustive search.
- Only one probing phase per MIN node, but it examines more than one child.

With probing factor 1, this is Star2. With probing factor 0 or N, it degrades
to Star1.

### Performance (branching factor 20, depth 3, average case):

| Probing factor | Cyclic Star2.5 | Sequential Star2.5 |
|----------------|----------------|---------------------|
| 1 (= Star2)   | 40.7%          | 40.7%               |
| 2              | 34.4%          | 36.8%               |
| 3              | 31.7%          | 34.7%               |
| 4              | 30.2%          | 34.7%               |
| 5              | 29.3%          | 36.6%  (degrading)  |
| 10             | 27.9%          | 50.4%               |
| 20 (= Star1)  | 27.9%          | 79.6%               |

**Observations:**

- Cyclic Star2.5 improves monotonically with probing factor, converging around
  probing factor ≈ N/2. After that, all possible cutoffs have already occurred.
- Sequential Star2.5 has a sweet spot around probing factor 3-4, then
  degrades because examining too many children per MIN node wastes effort on
  nodes that would have been cut off with fewer probes.
- Cyclic Star2.5 is better but has more overhead. The sequential variant is
  the practical choice.

---

## 7. Performance Comparison

### Best case (2-ply tree, percentage of leaves examined):

| Branching factor | Exhaustive | Star1 | Star2 |
|------------------|-----------|-------|-------|
| 4                | 100%      | 62.5% | 39.1% |
| 10               | 100%      | 67.0% | 16.6% |
| 20               | 100%      | 69.5% | 8.5%  |
| 40               | 100%      | 70.8% | 4.3%  |

### Average case (2-ply tree, random ordering):

| Branching factor | Star1  | Star2  |
|------------------|--------|--------|
| 4                | 84.1%  | 75.4%  |
| 10               | 81.1%  | 53.1%  |
| 20               | 79.9%  | 41.8%  |
| 40               | 78.8%  | 35.0%  |

### Deep trees (average case, branching factor 10):

| Depth | Star1  | Star2  |
|-------|--------|--------|
| 3     | 81%    | 57%    |
| 4     | 73%    | 46%    |
| 5     | 73%    | 22%    |
| 6     | —      | 12%    |
| 7     | —      | —      |

The advantage of Star2 grows dramatically with tree depth. At depth 5 with
branching factor 10, Star2 examines only 22% of leaves — a 78% reduction.

### Asymptotic complexity:

| Algorithm | Exhaustive | Star1 best | Star2 best |
|-----------|-----------|------------|------------|
| 2-ply tree| O(N³)     | 0.721·N³   | 1.721·N²   |

Star2 reduces the exponent by 1, analogous to how alpha-beta reduces a 2-ply
exhaustive search from O(N²) to O(N) in ordinary minimax trees.

---

## 8. Pseudocode

### 8.1 Star1 Procedure

Handles a \* node by evaluating children left-to-right with progressive bound
tightening.

```
function Star1(board, alpha, beta):
    determine N successors s[1], s[2], ..., s[N]
    if N == 0:
        return Terminal_Eval(board)

    A ← N * (alpha - U) + U         // derived alpha for child 1
    B ← N * (beta  - L) + L         // derived beta  for child 1
    vsum ← 0

    for i = 1 to N:
        AX ← max(A, L)              // clamp to valid range
        BX ← min(B, U)
        v  ← Eval(s[i], AX, BX)     // search child i with derived bounds

        if v ≤ A:                    // lower bound proves * ≤ alpha
            return alpha             // CUT OFF (alpha cutoff at * node)

        if v ≥ B:                    // upper bound proves * ≥ beta
            return beta              // CUT OFF (beta cutoff at * node)

        vsum ← vsum + v
        A ← A + U - v               // tighten alpha for next child
        B ← B + L - v               // tighten beta  for next child

    return vsum / N                  // no cutoff: return exact average
```

Where:
- `L` = global lower bound on any game value (e.g., minimum possible score)
- `U` = global upper bound on any game value (e.g., maximum possible score)
- `Eval(s, α, β)` calls `Max(s, α, β)` or `Min(s, α, β)` depending on node type

### 8.2 Star2 Procedure (for \* node followed by MIN nodes)

Two-phase search: quick probing first, then full search with tighter bounds.

```
function Star2Min(board, alpha, beta):
    determine N successors s[1], s[2], ..., s[N]
    if N == 0:
        return Terminal_Eval(board)

    A ← N * (alpha - U)             // initial A (before +U in loop)
    B ← N * (beta  - L)             // initial B (before +L in loop)
    BX ← min(B, U)

    // ── Phase 1: Probing ──────────────────────────────────
    // Evaluate ONE child of each MIN node to get quick upper bounds

    for i = 1 to N:
        A  ← A + U                  // would-be full bound (assumes rest = U)
        AX ← max(A, L)
        w[i] ← Probe(s[i], AX, BX) // evaluate just 1 child of MIN node s[i]

        if w[i] ≤ A:                // probe value proves * ≤ alpha
            return alpha             // CUT OFF during probing

        A ← A - w[i]                // replace U assumption with actual w[i]

    // ── Phase 2: Full search ──────────────────────────────
    // Probing didn't cut off. Search each MIN node fully, using
    // tighter A bounds derived from probe values.

    vsum ← 0
    for i = 1 to N:
        B  ← B + L
        A  ← A + w[i]               // restore probe value contribution
        AX ← max(A, L)
        BX ← min(B, U)
        v  ← Min(s[i], AX, BX)      // full search of MIN node

        if v ≤ A:
            return alpha             // CUT OFF during full search

        if v ≥ B:
            return beta

        vsum ← vsum + v
        A ← A - v                   // update with true value (replaces w[i])
        B ← B - v

    return vsum / N
```

Where `Probe(s, α, β)` picks one child of MIN node s (randomly or by
heuristic) and returns `Min(child, α, β)`. If s is a leaf, returns
`Terminal_Eval(s)`.

A symmetric `Star2Max` procedure handles \* nodes followed by MAX nodes, with
the roles of alpha/beta and min/max swapped.

### 8.3 Star2.5 Cyclic

Multiple rounds of probing before falling back to full search.

```
function Star2_5_Cyclic(board, alpha, beta, probing_factor):
    determine N successors s[1], ..., s[N]
    if N == 0:
        return Terminal_Eval(board)

    A ← N * (alpha - U)
    B ← N * (beta  - L)

    // Track which children of each MIN node have been probed
    probed_count[1..N] ← 0
    w[1..N] ← U                      // initial upper bound per MIN node

    // ── Probing rounds ────────────────────────────────────
    for round = 1 to probing_factor:
        for i = 1 to N:
            if probed_count[i] < (number of children of s[i]):
                pick next unprobed child c of s[i]
                val ← Eval(c, derived_alpha, derived_beta)
                w[i] ← min(w[i], val)     // tighten upper bound on MIN node
                probed_count[i] += 1

                // Recompute A using w[i] values
                recompute lower bound on * node
                if lower_bound ≥ beta:
                    return beta            // CUT OFF
                if upper_bound ≤ alpha:
                    return alpha           // CUT OFF

    // ── Full search phase ─────────────────────────────────
    // Same as Star2 Phase 2, with bounds tightened by all probes
    ...same as Star2 full search...
```

### 8.4 Star2.5 Sequential

Simpler: probe multiple children of each MIN node in a single pass, then
fall back.

```
function Star2_5_Seq(board, alpha, beta, probing_factor):
    determine N successors s[1], ..., s[N]
    if N == 0:
        return Terminal_Eval(board)

    A ← N * (alpha - U)
    B ← N * (beta  - L)

    // ── Probing phase ─────────────────────────────────────
    for i = 1 to N:
        // Probe 'probing_factor' children of MIN node s[i]
        w[i] ← U
        for k = 1 to min(probing_factor, children_of(s[i])):
            pick kth child c of s[i]
            val ← Eval(c, derived_alpha, derived_beta)
            w[i] ← min(w[i], val)

        // Update A using w[i] and check for cutoff
        A ← A + U
        if w[i] ≤ A:
            return alpha
        A ← A - w[i]

    // ── Full search phase ─────────────────────────────────
    // Same as Star2 Phase 2
    ...
```

### 8.5 The Standard Max and Min (for completeness)

```
function Max(board, alpha, beta):
    if board is terminal:
        return Terminal_Eval(board)

    determine successors s[1], ..., s[N]
    for i = 1 to N:
        v ← StarProcedure(s[i], alpha, beta)   // children are * nodes
        if v ≥ beta:
            return beta                          // beta cutoff
        alpha ← max(alpha, v)

    return alpha


function Min(board, alpha, beta):
    if board is terminal:
        return Terminal_Eval(board)

    determine successors s[1], ..., s[N]
    for i = 1 to N:
        v ← StarProcedure(s[i], alpha, beta)   // children are * nodes
        if v ≤ alpha:
            return alpha                         // alpha cutoff
        beta ← min(beta, v)

    return beta
```

In a regular \*-minimax tree, the call chain is:

```
Star → Min → Star → Max → Star → Min → ... → leaf
```

Each layer alternates between chance events (\*) and player decisions (+/-).

---

## References

Ballard, B.W. (1983). "The \*-Minimax Search Procedure for Trees Containing
Chance Nodes." *Artificial Intelligence*, 21, 327–350.

"""Measure and plot MCTS search depth as a function of game turn.

Supports two modes:
  --mode random   (default) Uses a lightweight random-policy with sequential
                  MCTS.  Fast, isolates depth vs branching.
  --mode nn       Uses the real neural network with the agent's batched MCTS.
                  Slower, but measures the actual tree depth during training.

Usage:
    python search_depth_analysis.py --num-search 20 50 100 200 --games 3
    python search_depth_analysis.py --mode nn --num-search 50 --games 1
    python search_depth_analysis.py --mode nn --num-search 50 --batch-size 8 --games 1
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import argparse
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from agents.mcts import MCTS, MCTSConfig, Node
from games.carcassonne_game import CarcassonneGame
from games.carcassonne_game_state import CarcassonneGameState


# ── tree statistics ──────────────────────────────────────────────────────

def tree_depth_stats(root: Node) -> dict:
    """Walk the MCTS tree from *root* and return depth statistics."""
    max_depth = 0
    weighted_depth_sum = 0.0
    weighted_count_sum = 0.0
    leaf_count = 0

    stack: List[Tuple[Node, int]] = [(root, 0)]
    while stack:
        node, d = stack.pop()
        if d > max_depth:
            max_depth = d

        has_visited_children = False
        if node.expanded and node.children:
            for child in node.children.values():
                if child.visit_count > 0:
                    stack.append((child, d + 1))
                    has_visited_children = True

        if not has_visited_children:
            visits = max(node.visit_count, 1)
            weighted_depth_sum += d * visits
            weighted_count_sum += visits
            leaf_count += 1

    node = root
    pv_depth = 0
    while node.expanded and node.children:
        best_child = max(node.children.values(), key=lambda n: n.visit_count)
        if best_child.visit_count == 0:
            break
        pv_depth += 1
        node = best_child

    mean_depth = weighted_depth_sum / max(weighted_count_sum, 1.0)
    return {
        "max_depth": max_depth,
        "pv_depth": pv_depth,
        "mean_depth": mean_depth,
        "num_leaves": leaf_count,
    }


def tree_node_count(root: Node) -> int:
    count = 0
    stack = [root]
    while stack:
        node = stack.pop()
        count += 1
        if node.expanded and node.children:
            for child in node.children.values():
                if child.visit_count > 0:
                    stack.append(child)
    return count


# ── random-policy model ──────────────────────────────────────────────────

def _random_model(state, legal_actions):
    n = len(legal_actions)
    return {"policy": {a: 1.0 / n for a in legal_actions}, "value": 0.0}


# ── game runners ─────────────────────────────────────────────────────────

def run_game_random(game, mcts_cfg: MCTSConfig, max_turns: int) -> List[dict]:
    """Play one game with random-policy sequential MCTS."""
    state = game.get_initial_state()
    records: List[dict] = []
    turn = 0

    while not state.is_terminal() and turn < max_turns:
        turn += 1
        branching = len(state.get_legal_actions())

        mcts = MCTS(_random_model, mcts_cfg)
        mcts.root = Node()
        mcts.root.state = state
        for _ in range(mcts_cfg.num_search):
            mcts._search(mcts.root)

        stats = tree_depth_stats(mcts.root)
        stats["turn"] = turn
        stats["branching"] = branching
        stats["tree_nodes"] = tree_node_count(mcts.root)
        records.append(stats)

        all_counts = {a: mcts.root.children[a].visit_count
                      for a in state.get_legal_actions()}
        total = sum(all_counts.values()) or 1
        actions = list(all_counts.keys())
        probs = np.array([all_counts[a] / total for a in actions])
        probs /= probs.sum()
        state = state.apply_action(np.random.choice(actions, p=probs))

        if turn % 20 == 0:
            sys.stdout.write(f"\r    turn {turn}...")
            sys.stdout.flush()

    if turn >= 20:
        sys.stdout.write("\r" + " " * 30 + "\r")
        sys.stdout.flush()
    return records


def run_game_nn(agent, game, max_turns: int, temperature_threshold: int) -> List[dict]:
    """Play one game using the agent's batched MCTS (with real NN)."""
    agent.reset_inference()
    state = game.get_initial_state()
    records: List[dict] = []
    turn = 0

    while not state.is_terminal() and turn < max_turns:
        turn += 1
        temp = (max(0.0, 1.0 - turn / temperature_threshold)
                if temperature_threshold > 0 else 0.0)
        branching = len(state.get_legal_actions())
        action_probs = agent.get_action_probs(
            state, temperature=temp, add_exploration_noise=True)

        stats = tree_depth_stats(agent.mcts.root)
        stats["turn"] = turn
        stats["branching"] = branching
        stats["tree_nodes"] = tree_node_count(agent.mcts.root)
        records.append(stats)

        actions = list(action_probs.keys())
        probs = np.array(list(action_probs.values()))
        probs /= probs.sum()
        state = state.apply_action(np.random.choice(actions, p=probs))

        if turn % 20 == 0:
            sys.stdout.write(f"\r    turn {turn}...")
            sys.stdout.flush()

    if turn >= 20:
        sys.stdout.write("\r" + " " * 30 + "\r")
        sys.stdout.flush()
    return records


# ── plotting ─────────────────────────────────────────────────────────────

COLORS = ["#e63946", "#457b9d", "#2a9d8f", "#f4a261", "#264653", "#8338ec"]


def make_plot(results_by_ns, args, mode_label):
    n_ns = len(results_by_ns)
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True,
                              gridspec_kw={"height_ratios": [3, 1, 1]})
    ns_list = sorted(results_by_ns.keys())
    title_ns = ", ".join(str(ns) for ns in ns_list)
    fig.suptitle(
        f"MCTS Search Depth ({mode_label})  |  "
        f"num_search={{{title_ns}}}, cpuct={args.cpuct}, "
        f"expl={args.root_exploration_fraction}, variant={args.variant}  |  "
        f"{args.games} games",
        fontsize=11, fontweight="bold")

    ax1, ax2, ax3 = axes

    for idx, ns in enumerate(ns_list):
        all_records = results_by_ns[ns]
        turns = sorted(all_records.keys())
        avg_max = np.array([np.mean(all_records[t]["max_depth"]) for t in turns])
        avg_pv = np.array([np.mean(all_records[t]["pv_depth"]) for t in turns])
        avg_mean = np.array([np.mean(all_records[t]["mean_depth"]) for t in turns])
        avg_branch = np.array([np.mean(all_records[t]["branching"]) for t in turns])
        avg_nodes = np.array([np.mean(all_records[t]["tree_nodes"]) for t in turns])
        std_pv = np.array([np.std(all_records[t]["pv_depth"]) for t in turns])

        c = COLORS[idx % len(COLORS)]
        ax1.plot(turns, avg_pv, label=f"PV depth (ns={ns})", color=c, lw=2.0)
        ax1.fill_between(turns, avg_pv - std_pv, avg_pv + std_pv,
                         color=c, alpha=0.10)
        ax1.plot(turns, avg_mean, color=c, ls="--", lw=1.0, alpha=0.6)

        if idx == 0:
            ax2.plot(turns, avg_branch, color="#6b705c", lw=1.2, alpha=0.8)
            ax2.fill_between(turns, 0, avg_branch, color="#6b705c", alpha=0.12)

        ax3.plot(turns, avg_nodes, label=f"ns={ns}", color=c, lw=1.2, alpha=0.8)

    ax1.set_ylabel("Depth (actions)")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)

    ax2.set_ylabel("Branching factor")
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=0)

    ax3.set_ylabel("Tree nodes (visited)")
    ax3.set_xlabel("Turn")
    ax3.legend(loc="upper left", fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(bottom=0)

    plt.tight_layout()
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {args.output}")
    plt.close(fig)


# ── printing ─────────────────────────────────────────────────────────────

def print_summary(all_records, ns):
    turns = sorted(all_records.keys())
    avg_max = [np.mean(all_records[t]["max_depth"]) for t in turns]
    avg_pv = [np.mean(all_records[t]["pv_depth"]) for t in turns]
    avg_mean = [np.mean(all_records[t]["mean_depth"]) for t in turns]
    avg_branch = [np.mean(all_records[t]["branching"]) for t in turns]

    print(f"\n  {'Turn':>5}  {'MaxD':>5}  {'PV_D':>5}  {'MeanD':>6}  {'Branch':>7}")
    print(f"  {'-'*38}")
    step = max(1, len(turns) // 15)
    for i in range(0, len(turns), step):
        t = turns[i]
        print(f"  {t:5d}  {avg_max[i]:5.1f}  {avg_pv[i]:5.1f}  "
              f"{avg_mean[i]:6.2f}  {avg_branch[i]:7.1f}")
    if turns and len(turns) > 1:
        i = len(turns) - 1
        t = turns[i]
        print(f"  {t:5d}  {avg_max[i]:5.1f}  {avg_pv[i]:5.1f}  "
              f"{avg_mean[i]:6.2f}  {avg_branch[i]:7.1f}")

    overall_max = np.mean([v for t in turns for v in all_records[t]["max_depth"]])
    overall_pv = np.mean([v for t in turns for v in all_records[t]["pv_depth"]])
    overall_mean = np.mean([v for t in turns for v in all_records[t]["mean_depth"]])
    overall_branch = np.mean([v for t in turns for v in all_records[t]["branching"]])
    print(f"\n  Overall: max_depth={overall_max:.2f}, pv_depth={overall_pv:.2f}, "
          f"mean_depth={overall_mean:.2f}, branching={overall_branch:.1f}")


# ── main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Measure MCTS search depth vs. game turn")
    parser.add_argument("--mode", default="random", choices=["random", "nn"],
                        help="'random' = fast random policy, 'nn' = real NN + batched MCTS")
    parser.add_argument("--num-search", type=int, nargs="+", default=[50])
    parser.add_argument("--cpuct", type=float, default=1.25)
    parser.add_argument("--root-exploration-fraction", type=float, default=0.25)
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size per round in multi-round batched MCTS")
    parser.add_argument("--variant", default="full", choices=["full", "base"])
    parser.add_argument("--games", type=int, default=3)
    parser.add_argument("--max-turns", type=int, default=9999)
    parser.add_argument("--model-type", default="gps", choices=["gps", "legacy"])
    parser.add_argument("--model-size", default="small")
    parser.add_argument("--temperature-threshold", type=int, default=140)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output", default="search_depth.png")
    args = parser.parse_args()

    ns_values = sorted(set(args.num_search))
    mode_label = "random policy, sequential" if args.mode == "random" else f"NN, batched (bs={args.batch_size})"
    print(f"Mode: {mode_label}")
    print(f"Config: num_search={ns_values}, cpuct={args.cpuct}, "
          f"root_expl={args.root_exploration_fraction}, variant={args.variant}, "
          f"games={args.games}")

    CarcassonneGameState.MARGIN_SCALE = 40.0
    game = CarcassonneGame(variant=args.variant)

    # Set up NN agent if needed (once, shared across ns values)
    agent = None
    if args.mode == "nn":
        import torch
        from agents.carcassonne_gnn_agent import CarcassonneGNNAgent
        from agents.mcts_ml_agent import MCTSMLAgentConfig

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {device}")

        mcts_cfg = MCTSConfig(
            num_search=ns_values[0],
            cpuct=args.cpuct,
            root_exploration_fraction=args.root_exploration_fraction,
            batch_size=args.batch_size,
        )
        agent_cfg = MCTSMLAgentConfig(mcts_config=mcts_cfg)
        dummy_writer = type("W", (), {"add_scalar": lambda *a, **k: None})()
        agent = CarcassonneGNNAgent(
            "depth_bench", summary_writer=dummy_writer,
            model_type=args.model_type, model_size=args.model_size,
            config=agent_cfg)
        agent.model_summarizer.finished = True
        agent.device = device
        agent.model = agent.model.to(device)

        if args.checkpoint:
            sd = torch.load(args.checkpoint, map_location=device)
            agent.model.load_state_dict(sd)
            print(f"Loaded checkpoint: {args.checkpoint}")
        else:
            print("Using random-init model (no checkpoint)")

    results_by_ns: Dict[int, Any] = {}

    for ns in ns_values:
        print(f"\n=== num_search = {ns} ===")

        all_records = defaultdict(lambda: {
            "max_depth": [], "pv_depth": [], "mean_depth": [],
            "branching": [], "tree_nodes": [],
        })

        for g in range(args.games):
            t0 = time.monotonic()

            if args.mode == "random":
                mcts_cfg = MCTSConfig(
                    num_search=ns, cpuct=args.cpuct,
                    root_exploration_fraction=args.root_exploration_fraction,
                    batch_size=args.batch_size,
                )
                records = run_game_random(game, mcts_cfg, args.max_turns)
            else:
                agent.config.mcts_config.num_search = ns
                agent.config.mcts_config.batch_size = args.batch_size
                records = run_game_nn(agent, game, args.max_turns,
                                      args.temperature_threshold)

            elapsed = time.monotonic() - t0
            n_turns = len(records)
            speed = n_turns / elapsed if elapsed > 0 else 0
            print(f"  Game {g+1}/{args.games}: {n_turns} turns in {elapsed:.1f}s "
                  f"({speed:.0f} turns/sec)")

            for rec in records:
                t = rec["turn"]
                for k in ("max_depth", "pv_depth", "mean_depth",
                          "branching", "tree_nodes"):
                    all_records[t][k].append(rec[k])

        print_summary(all_records, ns)
        results_by_ns[ns] = dict(all_records)

    make_plot(results_by_ns, args, mode_label)


if __name__ == "__main__":
    main()

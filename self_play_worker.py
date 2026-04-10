"""
Self-play worker functions for parallel training.

This module is deliberately kept free of Sacred, TensorBoard, and other
heavy framework imports so that spawned worker processes only load the
minimal dependencies (PyTorch, PyG, game engine, agent).
"""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

import itertools
import random
import time

import torch
import numpy as np

from agents.carcassonne_gnn_agent import CarcassonneGNNAgent
from agents.random_agent import RandomAgent
from game_stats import GameStats
from games.carcassonne_game import CarcassonneGame
from games.carcassonne_game_state import CarcassonneGameState

_worker_agent = None
_worker_random_agent = None
_worker_game = None
_worker_temp_threshold = None
_worker_games_per_worker = 1

_arena_agent1 = None
_arena_agent2 = None
_arena_game = None


class _NullWriter:
    """No-op replacement for TensorBoard SummaryWriter in worker processes."""
    def add_scalar(self, *a, **kw):
        pass


def init_worker(state_dict_cpu, model_type, model_size, temp_threshold,
                margin_scale, num_search=None, cpuct=1.25,
                root_exploration_fraction=0.25, game_variant="full",
                mcts_batch_size=8, games_per_worker=1):
    """Called once per worker process to create an agent (GPU if available, else CPU)."""
    global _worker_agent, _worker_random_agent, _worker_game, _worker_temp_threshold, _worker_games_per_worker
    pid = os.getpid()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  [Worker {pid}] Initializing (model_type={model_type}, model_size={model_size}, device={device})...", flush=True)
    t0 = time.monotonic()
    CarcassonneGameState.MARGIN_SCALE = margin_scale
    _worker_temp_threshold = temp_threshold
    _worker_game = CarcassonneGame(variant=game_variant)
    _worker_random_agent = RandomAgent()

    config = None
    if num_search is not None:
        from agents.mcts_ml_agent import MCTSMLAgentConfig
        from agents.mcts import MCTSConfig
        config = MCTSMLAgentConfig(mcts_config=MCTSConfig(
            num_search=num_search, cpuct=cpuct,
            root_exploration_fraction=root_exploration_fraction,
            batch_size=mcts_batch_size,
        ))

    _worker_agent = CarcassonneGNNAgent(
        'worker', summary_writer=_NullWriter(), model_type=model_type,
        model_size=model_size, config=config)
    _worker_agent.model_summarizer.finished = True
    _worker_agent.device = device
    _worker_agent.model = _worker_agent.model.to(device)
    _worker_agent.model.load_state_dict(state_dict_cpu)
    _worker_agent.reset_inference()
    _worker_games_per_worker = games_per_worker
    print(f"  [Worker {pid}] Ready on {device}, games_per_worker={games_per_worker} "
          f"({time.monotonic() - t0:.1f}s)", flush=True)


def run_episode(args):
    """Run one self-play episode.

    Returns (data, p0_score, p1_score, turns, elapsed, stats_line).
    args: (episode_idx, vs_random_bool)
    """
    _idx, vs_random = args
    pid = os.getpid()
    tag = "vs-random" if vs_random else "self-play"
    #print(f"  [Worker {pid}] Episode {_idx} ({tag}): starting", flush=True)
    t0 = time.monotonic()
    _worker_agent.reset_inference()
    state = _worker_game.get_initial_state()
    history = []
    gstats = GameStats()
    turn = 0
    _next_log_turn = 40

    random_seat = random.randint(0, 1) if vs_random else -1

    for _ in itertools.count():
        turn += 1
        if turn == _next_log_turn:
            elapsed_so_far = time.monotonic() - t0
            #print(f"  [Worker {pid}] Episode {_idx}: turn {turn}/~180, "
            #      f"scores={state.get_player_score(0)}-{state.get_player_score(1)}, "
            #      f"{elapsed_so_far:.0f}s elapsed", flush=True)
            _next_log_turn += 40

        current_player = state.get_current_player()

        if vs_random and current_player == random_seat:
            action_probs = _worker_random_agent.get_action_probs(
                state, temperature=1, add_exploration_noise=False)
        else:
            temperature = max(0.0, 1.0 - turn / _worker_temp_threshold) if _worker_temp_threshold > 0 else 0.0
            action_probs = _worker_agent.get_action_probs(
                state, temperature=temperature, add_exploration_noise=True)
            history.append((state, action_probs, current_player))

        actions = list(action_probs.keys())
        probs = np.array([action_probs[a] for a in actions])
        probs /= probs.sum()
        action = np.random.choice(actions, p=probs)
        gstats.record_action(current_player, action, pre_state=state)
        state = state.apply_action(action)
        gstats.record_score_delta(state)
        if state.is_terminal():
            value = state.get_player_value(state.get_current_player())
            cp = state.get_current_player()
            data = [(s, p, value * ((-1) ** (cp != player)))
                    for s, p, player in history]
            gstats.set_final_scores(state)
            elapsed = time.monotonic() - t0
            stats_line = gstats.summary_line(f"[Worker {pid}] Ep {_idx} ({tag})")
            print(f"  {stats_line}", flush=True)
            return (data, state.get_player_score(0), state.get_player_score(1),
                    turn, elapsed, stats_line)


# ── Cross-game batched self-play ──────────────────────────────────────────────

def run_multi_episode(args):
    """Run *games_per_worker* self-play episodes concurrently in one process.

    All games share a single model and their MCTS leaf evaluations are
    merged into one large NN batch per round, maximising GPU utilisation.

    args: list of (episode_idx, vs_random_bool) tuples.
    Returns: list of (data, p0_score, p1_score, turns, elapsed, stats_line)
             in the same order as *args*.
    """
    pid = os.getpid()
    t0 = time.monotonic()

    # ── Initialise per-game state ────────────────────────────────────────
    n_games = len(args)
    games = []
    for ep_idx, vs_random in args:
        tag = "vs-random" if vs_random else "self-play"
        random_seat = random.randint(0, 1) if vs_random else -1
        _worker_agent.reset_inference()
        mcts = _worker_agent.mcts
        state = _worker_game.get_initial_state()
        games.append({
            "ep_idx": ep_idx,
            "vs_random": vs_random,
            "tag": tag,
            "random_seat": random_seat,
            "state": state,
            "mcts": mcts,
            "history": [],
            "gstats": GameStats(),
            "turn": 0,
            "finished": False,
            "result": None,
            "awaiting_search": True,
        })

    results_ordered = [None] * n_games

    while True:
        active = [g for g in games if not g["finished"]]
        if not active:
            break

        # ── Advance each active game: decide action or start search ──────
        needs_search = []
        for g in active:
            if g["awaiting_search"]:
                state = g["state"]
                g["turn"] += 1
                turn = g["turn"]
                cp = state.get_current_player()

                if g["vs_random"] and cp == g["random_seat"]:
                    action_probs = _worker_random_agent.get_action_probs(
                        state, temperature=1, add_exploration_noise=False)
                    _apply_action(g, action_probs, record_history=False)
                    continue

                temp = (max(0.0, 1.0 - turn / _worker_temp_threshold)
                        if _worker_temp_threshold > 0 else 0.0)
                mcts = _worker_agent._prepare_mcts()
                g["mcts"] = mcts
                g["temperature"] = temp
                mcts.prepare_root(state, add_exploration_noise=True)
                needs_search.append(g)

        # ── Multi-round batched search across all active games ───────────
        if needs_search:
            _run_cross_game_search(needs_search)

            for g in needs_search:
                action_probs = g["mcts"].extract_action_probs(
                    g["state"], g["temperature"])
                g["history"].append(
                    (g["state"], action_probs, g["state"].get_current_player()))
                _apply_action(g, action_probs, record_history=False)

    elapsed = time.monotonic() - t0
    return [g["result"] for g in games]


def _run_cross_game_search(games_needing_search):
    """Run multi-round MCTS for all games, batching NN calls across them."""
    while True:
        all_pending = []
        for g in games_needing_search:
            mcts = g["mcts"]
            if mcts.search_done:
                continue
            pending = mcts.collect_pending_leaves()
            for item in pending:
                all_pending.append((g, item))

        if not all_pending:
            break

        states_and_actions = [
            (item[2], item[3]) for _, item in all_pending
        ]
        results = _worker_agent.batched_evaluate(states_and_actions)

        idx = 0
        game_items = {}
        for g, item in all_pending:
            gid = id(g)
            if gid not in game_items:
                game_items[gid] = (g["mcts"], [], [])
            game_items[gid][1].append(item)
            game_items[gid][2].append(results[idx])
            idx += 1

        for mcts, items, res_list in game_items.values():
            mcts.deliver_results(items, res_list)


def _apply_action(g, action_probs, record_history=True):
    """Pick an action from probs, apply it, check terminal."""
    state = g["state"]
    if record_history:
        g["history"].append(
            (state, action_probs, state.get_current_player()))

    actions = list(action_probs.keys())
    probs = np.array([action_probs[a] for a in actions])
    probs /= probs.sum()
    action = np.random.choice(actions, p=probs)

    cp = state.get_current_player()
    g["gstats"].record_action(cp, action, pre_state=state)
    new_state = state.apply_action(action)
    g["gstats"].record_score_delta(new_state)
    g["state"] = new_state

    if new_state.is_terminal():
        _finish_game(g)
    else:
        g["awaiting_search"] = True


def _finish_game(g):
    """Finalise a completed game."""
    pid = os.getpid()
    state = g["state"]
    value = state.get_player_value(state.get_current_player())
    cp = state.get_current_player()
    data = [(s, p, value * ((-1) ** (cp != player)))
            for s, p, player in g["history"]]
    g["gstats"].set_final_scores(state)
    stats_line = g["gstats"].summary_line(
        f"[Worker {pid}] Ep {g['ep_idx']} ({g['tag']})")
    print(f"  {stats_line}", flush=True)
    g["result"] = (data, state.get_player_score(0), state.get_player_score(1),
                   g["turn"], 0.0, stats_line)
    g["finished"] = True
    g["awaiting_search"] = False


# ── Arena worker functions ────────────────────────────────────────────────────

def init_arena_worker(state_dict_agent1, state_dict_agent2,
                      model_type, model_size, margin_scale,
                      num_search, cpuct, game_variant="full",
                      mcts_batch_size=8):
    """Initialise an arena worker with two model agents (or one model + random).

    If *state_dict_agent2* is ``None`` the second player is a ``RandomAgent``.
    """
    global _arena_agent1, _arena_agent2, _arena_game
    pid = os.getpid()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    t0 = time.monotonic()
    CarcassonneGameState.MARGIN_SCALE = margin_scale
    _arena_game = CarcassonneGame(variant=game_variant)

    from agents.mcts_ml_agent import MCTSMLAgentConfig
    from agents.mcts import MCTSConfig
    cfg = MCTSMLAgentConfig(mcts_config=MCTSConfig(
        num_search=num_search, cpuct=cpuct, batch_size=mcts_batch_size))

    _arena_agent1 = CarcassonneGNNAgent(
        'arena1', summary_writer=_NullWriter(), model_type=model_type,
        model_size=model_size, config=cfg)
    _arena_agent1.model_summarizer.finished = True
    _arena_agent1.device = device
    _arena_agent1.model = _arena_agent1.model.to(device)
    _arena_agent1.model.load_state_dict(state_dict_agent1)
    _arena_agent1.reset_inference()

    if state_dict_agent2 is not None:
        _arena_agent2 = CarcassonneGNNAgent(
            'arena2', summary_writer=_NullWriter(), model_type=model_type,
            model_size=model_size, config=cfg)
        _arena_agent2.model_summarizer.finished = True
        _arena_agent2.device = device
        _arena_agent2.model = _arena_agent2.model.to(device)
        _arena_agent2.model.load_state_dict(state_dict_agent2)
        _arena_agent2.reset_inference()
    else:
        _arena_agent2 = RandomAgent()

    print(f"  [ArenaWorker {pid}] Ready on {device} ({time.monotonic() - t0:.1f}s)", flush=True)


def run_arena_game(args):
    """Play one arena game.

    args: (game_idx, swap_seats_bool)
        swap_seats: if True, agent2 plays seat 0 and agent1 plays seat 1.

    Returns: (winner_from_agent1_perspective, agent1_score, agent2_score, stats_line)
        winner: 1 = agent1 wins, -1 = agent2 wins, 0 = draw
    """
    game_idx, swap_seats = args
    pid = os.getpid()

    if swap_seats:
        players = [_arena_agent2, _arena_agent1]
    else:
        players = [_arena_agent1, _arena_agent2]

    for p in players:
        p.reset_inference()

    state = _arena_game.get_initial_state()
    gstats = GameStats()
    for _ in itertools.count():
        if state.is_terminal():
            break
        cp = state.get_current_player()
        action_probs = players[cp].get_action_probs(
            state, temperature=0, add_exploration_noise=False)
        actions = list(action_probs.keys())
        probs = np.array([action_probs[a] for a in actions])
        probs /= probs.sum()
        action = np.random.choice(actions, p=probs)
        gstats.record_action(cp, action, pre_state=state)
        state = state.apply_action(action)
        gstats.record_score_delta(state)

    scores = [state.get_player_score(i) for i in range(state.get_num_players())]
    values = [state.get_player_value(i) for i in range(state.get_num_players())]
    gstats.set_final_scores(state)

    if swap_seats:
        a1_score, a2_score = scores[1], scores[0]
        a1_val, a2_val = values[1], values[0]
    else:
        a1_score, a2_score = scores[0], scores[1]
        a1_val, a2_val = values[0], values[1]

    if a1_val > a2_val:
        result = 1
    elif a1_val < a2_val:
        result = -1
    else:
        result = 0

    res_tag = 'a1_win' if result == 1 else 'a2_win' if result == -1 else 'draw'
    stats_line = gstats.summary_line(
        f"[ArenaWorker {pid}] Game {game_idx} ({res_tag})")
    print(f"  {stats_line}", flush=True)
    return (result, a1_score, a2_score, stats_line)


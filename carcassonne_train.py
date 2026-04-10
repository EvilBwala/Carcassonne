import os as _os
_os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

import warnings
warnings.filterwarnings('ignore', message='.*tee_std.*timeout.*')

from collections import deque
import itertools
import json
import math
import os
import random
import sys
from typing import Any, Dict, List
import torch
import torch.multiprocessing as _mp
import numpy as np
from pathlib import Path

from datetime import datetime
import time

from sacred import Experiment
from sacred.observers import FileStorageObserver
from agents.agent import Agent
from agents.carcassonne_agent import CarcassonneAgent
from agents.carcassonne_gnn_agent import CarcassonneGNNAgent
from agents.ml_agent import MLAgent
from agents.random_agent import RandomAgent
try:
    from carcassonne_engine import action_from_json, action_to_json, FarmUtil
except ImportError:
    from carcassonne.carcassonne_game_state import action_from_json, action_to_json
    from carcassonne.utils.farm_util import FarmUtil
from games.carcassonne_game import CarcassonneGame
from games.carcassonne_game_state import CarcassonneGameState
from games.game import Game
from games.game_state import GameState

from utils.context import Context, default, factory, patch
from utils.experiment import ex, init_experiment
from utils.utils import CheckpointDirManager, CyclicCounter, CheckpointManager, ModelSummarizer, argmax_dict, pretty_print_dict
from utils.debug import debug
from game_stats import GameStats
from self_play_worker import (
    init_worker as _init_sp_worker, run_episode as _run_sp_episode,
    run_multi_episode as _run_multi_episode,
    init_arena_worker as _init_arena_worker, run_arena_game as _run_arena_game,
)

from tqdm import tqdm


NAME = 'carcassonne'
init_experiment(NAME, Experiment(NAME))


def _log(msg: str):
    ts = datetime.now().strftime('%H:%M:%S')
    print(f"[{ts}] {msg}", flush=True)


@ex.sacred.config
def config():
    load_checkpoint_path = None
    num_iters = 200
    temperature_threshold = 120
    arena_iterations = 0
    arena_win_threshold = 0.0
    estimated_turns_per_game = 144
    random_arena_iterations = 50
    model_type = 'gps'
    model_size = 'small'
    margin_scale = 10.0
    num_workers = 20

    # ── Replay buffer ─────────────────────────────────────────────────
    replay_buffer_size = 25000       # max (state, policy, value) tuples kept

    # ── SGD optimizer (AlphaZero-style) ───────────────────────────────
    learning_rate = 0.1             # SGD initial LR (higher than Adam)
    momentum = 0.9
    weight_decay = 1e-4
    gradient_clipping = 2.0
    lr_milestones = [100, 150]         # iterations at which to decay LR
    lr_gamma = 0.5                   # LR multiplied by this at each milestone

    # ── Training steps (replaces epoch-based training) ────────────────
    train_steps_per_iter = 100       # gradient steps per iteration
    train_batch_size = 256           # mini-batch size (sampled from replay buffer)

    # ── MCTS exploration ─────────────────────────────────────────────
    root_exploration_fraction = 0.25  # fraction of root prior from Dirichlet noise
    random_opponent_fraction = 0.2    # fraction of self-play games vs random opponent

    # ── MCTS batching ──────────────────────────────────────────────────
    mcts_batch_size = 16              # leaves per round in multi-round batched MCTS
    games_per_worker = 1              # games run concurrently per worker (cross-game batching)

    # ── Scheduled parameters (init → final, linearly ramped) ──────────
    schedule_ramp_iters = 0          # 0 = ramp over num_iters
    num_self_play_episodes_init = 40
    num_self_play_episodes_final = 80
    num_search_init = 100
    num_search_final = 200
    cpuct_init = 2.0
    cpuct_final = 1.25

    # ── Game variant ───────────────────────────────────────────────────
    game_variant = "base"               # "full" (I&C + abbots) or "base"


@ex.sacred.automain
def run(experiment_dir, tensorboard_dir, load_checkpoint_path, num_iters,
        temperature_threshold,
        arena_iterations, arena_win_threshold,
        estimated_turns_per_game, random_arena_iterations, model_type,
        model_size, margin_scale, num_workers,
        replay_buffer_size,
        learning_rate, momentum, weight_decay, gradient_clipping,
        lr_milestones, lr_gamma,
        train_steps_per_iter, train_batch_size,
        root_exploration_fraction, random_opponent_fraction,
        mcts_batch_size, games_per_worker,
        schedule_ramp_iters,
        num_self_play_episodes_init, num_self_play_episodes_final,
        num_search_init, num_search_final,
        cpuct_init, cpuct_final,
        game_variant):
    from agents.mcts_ml_agent import MCTSMLAgentConfig
    from agents.mcts import MCTSConfig
    CarcassonneGameState.MARGIN_SCALE = margin_scale
    _log("Initializing agents and game...")

    ramp_total = schedule_ramp_iters if schedule_ramp_iters > 0 else num_iters

    def _schedule(init_val, final_val, iteration):
        """Linear ramp from init_val to final_val over ramp_total iterations."""
        frac = min(iteration / max(ramp_total - 1, 1), 1.0)
        return init_val + frac * (final_val - init_val)

    mcts_cfg = MCTSConfig(cpuct=cpuct_init, root_exploration_fraction=root_exploration_fraction,
                          batch_size=mcts_batch_size)
    agent_config = MCTSMLAgentConfig(
        learning_rate=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
        gradient_clipping=gradient_clipping,
        lr_milestones=tuple(lr_milestones),
        lr_gamma=lr_gamma,
        train_steps_per_iter=train_steps_per_iter,
        train_batch_size=train_batch_size,
        mcts_config=mcts_cfg,
    )
    prev_agent: MLAgent = CarcassonneGNNAgent('prev_agent', summary_writer=ex.summary_writer,
                                               config=agent_config,
                                               model_type=model_type, model_size=model_size)
    prev_agent.model_summarizer.finished = True
    agent: MLAgent = CarcassonneGNNAgent('new_agent', summary_writer=ex.summary_writer,
                                          config=agent_config,
                                          model_type=model_type, model_size=model_size)
    random_agent: Agent = RandomAgent()
    game: Game = CarcassonneGame(variant=game_variant)
    if game_variant == "base" and estimated_turns_per_game == 180:
        estimated_turns_per_game = 144
    _log(f"Agents initialized (model_type={model_type}, model_size={model_size}, "
         f"num_workers={num_workers}, game_variant={game_variant})")
    _log(f"  SGD: lr={learning_rate}, momentum={momentum}, wd={weight_decay}, "
         f"clip={gradient_clipping}, milestones={lr_milestones}, gamma={lr_gamma}")
    _log(f"  Training: {train_steps_per_iter} steps/iter, batch={train_batch_size}, "
         f"buffer={replay_buffer_size}")
    _log(f"  Exploration: root_noise_frac={root_exploration_fraction}, "
         f"random_opponent_frac={random_opponent_fraction}")
    _log(f"  Schedules: episodes={num_self_play_episodes_init}->{num_self_play_episodes_final}, "
         f"num_search={num_search_init}->{num_search_final}, "
         f"cpuct={cpuct_init}->{cpuct_final}, ramp={ramp_total} iters")

    replay_buffer: deque = deque(maxlen=replay_buffer_size)

    temp_save_path = os.path.join(experiment_dir, f'temp_checkpoint')
    checkpoints_dir = os.path.join(experiment_dir, 'checkpoints')
    checkpoints_manager = CheckpointDirManager(checkpoints_dir)

    current_iteration_start = 0

    def replay_buffer_from_json(json_data: List[Dict[str, Any]]):
        buf = deque(maxlen=replay_buffer_size)
        for entry in tqdm(json_data, desc="Loading replay buffer", leave=False, mininterval=1):
            state = CarcassonneGameState.from_json(entry['state'])
            action_probs = {}
            for action_json_data, prob in entry['action_probs']:
                action = action_from_json(action_json_data)
                action_probs[action] = prob
            buf.append((state, action_probs, entry['value']))
        return buf

    def replay_buffer_to_json(buf):
        json_data = []
        for state, action_probs, value in buf:
            json_data.append({
                'state': state.to_json(),
                'action_probs': [(action_to_json(action), prob) for action, prob in action_probs.items()],
                'value': value,
            })
        return json_data

    def execute_episode(metrics_step, vs_random=False):
        i = 0
        state = game.get_initial_state()
        train_data = []
        gstats = GameStats()
        agent.reset_inference()

        random_seat = random.randint(0, 1) if vs_random else -1
        tag = "vs-random" if vs_random else "self-play"

        for _ in tqdm(itertools.count(), desc="Self play (episode)", total=estimated_turns_per_game, leave=False, mininterval=1, disable=False):
            i += 1
            current_player = state.get_current_player()

            if vs_random and current_player == random_seat:
                action_probs = random_agent.get_action_probs(state, temperature=1, add_exploration_noise=False)
            else:
                temperature = max(0.0, 1.0 - i / temperature_threshold) if temperature_threshold > 0 else 0.0
                action_probs = agent.get_action_probs(state, temperature=temperature, add_exploration_noise=True)
                train_data.append((state, action_probs, current_player))

            actions = list(action_probs.keys())
            probs = np.array([action_probs[a] for a in actions])
            probs /= probs.sum()
            action = np.random.choice(actions, p=probs)
            gstats.record_action(current_player, action, pre_state=state)
            state = state.apply_action(action)
            gstats.record_score_delta(state)

            if state.is_terminal():
                value = state.get_player_value(state.get_current_player())
                data = [(s, p, value * ((-1) ** (state.get_current_player() != cp)))
                        for s, p, cp in train_data]

                gstats.set_final_scores(state)
                _log(gstats.summary_line(f"Self-play ({tag})"))

                player_1_score = state.get_player_score(0)
                player_2_score = state.get_player_score(1)
                ex.summary_writer.add_scalar('self_play_episode_player_1_score', player_1_score, global_step=metrics_step)
                ex.summary_writer.add_scalar('self_play_episode_player_2_score', player_2_score, global_step=metrics_step)

                if debug.self_play:
                    train_data[-2][0].visualize()
                    input()

                return data

        raise Exception("Not possible")

    def play_arena_games_parallel(total_games, state_dict_a1, state_dict_a2,
                                   label="Arena"):
        """Run *total_games* arena games across *num_workers* processes.

        Each agent pair plays half the games as seat 0 and half as seat 1.
        state_dict_a2 may be None, in which case agent2 is a RandomAgent.

        Returns: (a1_wins, a2_wins, draws, a1_avg_score, a2_avg_score)
        """
        half = total_games // 2
        game_args = [(i, False) for i in range(half)] + \
                    [(i + half, True) for i in range(total_games - half)]

        ctx = _mp.get_context('spawn')
        with ctx.Pool(
            num_workers,
            initializer=_init_arena_worker,
            initargs=(state_dict_a1, state_dict_a2,
                      model_type, model_size, margin_scale, num_search, cpuct,
                      game_variant, mcts_batch_size),
        ) as pool:
            results = list(pool.imap_unordered(_run_arena_game, game_args))

        a1_wins = sum(1 for r in results if r[0] == 1)
        a2_wins = sum(1 for r in results if r[0] == -1)
        draws   = sum(1 for r in results if r[0] == 0)
        a1_score = sum(r[1] for r in results) / max(len(results), 1)
        a2_score = sum(r[2] for r in results) / max(len(results), 1)
        return a1_wins, a2_wins, draws, a1_score, a2_score

    def play_arena_games_sequential(game: Game, total_games, agent1: Agent, agent2: Agent):
        """Fallback sequential arena for num_workers=1 or debug mode."""
        half = total_games // 2
        win1 = 0
        win2 = 0
        draws = 0

        def play_game(players: List[Agent], label: str = ""):
            state: GameState = game.get_initial_state()
            gstats = GameStats()
            for player in players:
                player.reset_inference()

            for _ in tqdm(itertools.count(), desc="Arena game", total=estimated_turns_per_game, leave=False, mininterval=1):
                if debug.arena:
                    print("[debug] state.get_current_player() = {}".format(state.get_current_player()))
                    print('player 1 score:', state.lib_state.scores[0])  # type: ignore
                    print('player 2 score:', state.lib_state.scores[1])  # type: ignore
                    print('player 1 meeples', state.lib_state.meeples[0])  # type: ignore
                    print('player 2 meeples', state.lib_state.meeples[1])  # type: ignore
                    if not state.is_terminal():
                        print('allowed actions: ', state.get_legal_actions())
                    state.visualize()
                    input()

                if state.is_terminal():
                    break

                cp = state.get_current_player()
                action_probs = players[cp].get_action_probs(state, temperature=0, add_exploration_noise=False)
                actions = list(action_probs.keys())
                probs = np.array([action_probs[a] for a in actions])
                probs /= probs.sum()
                action = random.choices(actions, weights=probs.tolist())[0]
                gstats.record_action(cp, action, pre_state=state)
                if debug.arena:
                    print("[debug] action = {}".format(action))

                state = state.apply_action(action)
                gstats.record_score_delta(state)

            gstats.set_final_scores(state)
            _log(gstats.summary_line(label))

            values = [state.get_player_value(i) for i in range(state.get_num_players())]
            if debug.arena:
                print("[debug] values = {}".format(values))
            max_value = max(values)
            winners = [i for i, v in enumerate(values) if v == max_value]
            winner = winners[0] if len(winners) == 1 else -1
            if debug.arena:
                print("[debug] winner = {}".format(winner))
            scores = [state.get_player_score(i) for i in range(state.get_num_players())]
            return winner, scores

        score1_sum = 0
        score2_sum = 0

        for gi in tqdm(range(half), desc="Arena.playGames (1)", mininterval=1, leave=False):
            result, scores = play_game([agent1, agent2], label=f"Arena(a1=P0) game {gi}")
            if result == 0:
                win1 += 1
            elif result == 1:
                win2 += 1
            else:
                draws += 1
            score1_sum += scores[0]
            score2_sum += scores[1]

        for gi in tqdm(range(total_games - half), desc="Arena.playGames (2)", mininterval=1, leave=False):
            result, scores = play_game([agent2, agent1], label=f"Arena(a1=P1) game {gi}")
            if result == 1:
                win1 += 1
            elif result == 0:
                win2 += 1
            else:
                draws += 1
            score1_sum += scores[1]
            score2_sum += scores[0]

        score1 = score1_sum / max(total_games, 1)
        score2 = score2_sum / max(total_games, 1)

        return win1, win2, draws, score1, score2

    def save():
        path = checkpoints_manager.next_checkpoint_path(current_iteration)
        best_model_path = os.path.join(path, 'best')
        os.makedirs(best_model_path, exist_ok=True)
        agent.save(best_model_path)
        training_state_path = os.path.join(path, 'training_state.json')
        with open(training_state_path, 'w') as f:
            json.dump({
                'current_iteration': current_iteration,
            }, f)
        training_data_path = os.path.join(path, 'training_data.json')
        with open(training_data_path, 'w') as f:
            json.dump(replay_buffer_to_json(replay_buffer), f)

    def load(path):
        nonlocal current_iteration_start, replay_buffer
        best_model_path = os.path.join(path, 'best')
        agent.load(best_model_path)
        training_state_path = os.path.join(path, 'training_state.json')
        with open(training_state_path, 'r') as f:
            training_state = json.load(f)
            current_iteration_start = training_state['current_iteration'] + 1

        train_data_path = os.path.join(path, 'training_data.json')
        if os.path.exists(train_data_path):
            with open(train_data_path, 'r') as f:
                print('Loading replay buffer...')
                replay_buffer = replay_buffer_from_json(json.load(f))

    if load_checkpoint_path is not None:
        _log(f"Loading checkpoint from: {load_checkpoint_path}")
        load(load_checkpoint_path)
        _log("Checkpoint loaded successfully")

    # ── Warm-up: fill replay buffer before training begins ─────────────
    if len(replay_buffer) < replay_buffer_size:
        shortfall = replay_buffer_size - len(replay_buffer)
        samples_per_game = max(estimated_turns_per_game // 2, 1)
        warmup_games = int(math.ceil(shortfall / samples_per_game))

        _log(f"WARM-UP: replay buffer {len(replay_buffer)}/{replay_buffer_size} — "
             f"playing ~{warmup_games} games to fill "
             f"(est. {samples_per_game} samples/game)")

        warmup_num_search = num_search_init
        warmup_cpuct = cpuct_init
        agent.config.mcts_config.num_search = warmup_num_search
        agent.config.mcts_config.cpuct = warmup_cpuct

        warmup_t0 = time.monotonic()
        warmup_samples: List = []

        if num_workers > 1:
            state_dict_cpu = {k: v.cpu() for k, v in agent.model.state_dict().items()}
            ctx = _mp.get_context('spawn')
            n_random_warmup = int(warmup_games * random_opponent_fraction)
            ep_args = [(i, i < n_random_warmup) for i in range(warmup_games)]
            random.shuffle(ep_args)

            sp_initargs = (state_dict_cpu, model_type, model_size,
                           temperature_threshold, margin_scale,
                           warmup_num_search, warmup_cpuct,
                           root_exploration_fraction, game_variant,
                           mcts_batch_size, games_per_worker)

            if games_per_worker > 1:
                gpw = games_per_worker
                batched_args = [ep_args[i:i+gpw]
                                for i in range(0, len(ep_args), gpw)]
                effective_workers = min(num_workers, len(batched_args))
                with ctx.Pool(effective_workers, initializer=_init_sp_worker,
                              initargs=sp_initargs) as pool:
                    done = 0
                    for multi_result in pool.imap_unordered(
                            _run_multi_episode, batched_args):
                        for data, p0, p1, turns, elapsed, _stats in multi_result:
                            warmup_samples.extend(data)
                            done += 1
                        if done % 20 == 0 or done >= warmup_games:
                            _log(f"  Warm-up: {done}/{warmup_games} games, "
                                 f"{len(warmup_samples)} samples so far")
            else:
                with ctx.Pool(num_workers, initializer=_init_sp_worker,
                              initargs=sp_initargs) as pool:
                    done = 0
                    for result in pool.imap_unordered(_run_sp_episode, ep_args):
                        data, p0, p1, turns, elapsed, _stats = result
                        warmup_samples.extend(data)
                        done += 1
                        if done % 20 == 0 or done == warmup_games:
                            _log(f"  Warm-up: {done}/{warmup_games} games, "
                                 f"{len(warmup_samples)} samples so far")
        else:
            n_random_warmup = int(warmup_games * random_opponent_fraction)
            episode_types = [True] * n_random_warmup + [False] * (warmup_games - n_random_warmup)
            random.shuffle(episode_types)
            for i in tqdm(range(warmup_games), desc="Warm-up self-play"):
                warmup_samples.extend(execute_episode(i, vs_random=episode_types[i]))
                if (i + 1) % 20 == 0 or i + 1 == warmup_games:
                    _log(f"  Warm-up: {i+1}/{warmup_games} games, "
                         f"{len(warmup_samples)} samples so far")

        replay_buffer.extend(warmup_samples)
        warmup_elapsed = time.monotonic() - warmup_t0
        _log(f"WARM-UP COMPLETE: {len(warmup_samples)} samples from {warmup_games} games "
             f"in {warmup_elapsed:.1f}s ({warmup_elapsed/60:.1f}min). "
             f"Buffer: {len(replay_buffer)}/{replay_buffer_size}")

    _log(f"Starting training: {num_iters} iterations")
    for current_iteration in tqdm(range(current_iteration_start, current_iteration_start + num_iters), desc="Iterations"):
        iter_t0 = time.monotonic()
        _log(f"===== ITERATION {current_iteration} / {current_iteration_start + num_iters - 1} =====")

        # --- Per-iteration parameter schedule ---
        num_search   = int(round(_schedule(num_search_init, num_search_final, current_iteration)))
        num_episodes = int(round(_schedule(num_self_play_episodes_init, num_self_play_episodes_final, current_iteration)))
        cpuct        = _schedule(cpuct_init, cpuct_final, current_iteration)

        agent.config.mcts_config.num_search = num_search
        agent.config.mcts_config.cpuct = cpuct
        prev_agent.config.mcts_config.num_search = num_search
        prev_agent.config.mcts_config.cpuct = cpuct

        _log(f"[Iter {current_iteration}] Schedule: episodes={num_episodes}, "
             f"num_search={num_search}, cpuct={cpuct:.3f}")

        # --- Phase 1: Self-play ---
        phase_t0 = time.monotonic()
        _log(f"[Iter {current_iteration}] PHASE 1/5: Self-play ({num_episodes} episodes)")
        new_samples = []

        if num_workers > 1:
            _log(f"[Iter {current_iteration}] Copying model weights to CPU for transfer to workers...")
            state_dict_cpu = {k: v.cpu() for k, v in agent.model.state_dict().items()}
            ctx = _mp.get_context('spawn')
            n_random_par = int(num_episodes * random_opponent_fraction)
            ep_args = [(i, i < n_random_par) for i in range(num_episodes)]
            random.shuffle(ep_args)

            sp_initargs = (state_dict_cpu, model_type, model_size,
                           temperature_threshold, margin_scale, num_search, cpuct,
                           root_exploration_fraction, game_variant,
                           mcts_batch_size, games_per_worker)

            results = []
            if games_per_worker > 1:
                gpw = games_per_worker
                batched_args = [ep_args[i:i+gpw]
                                for i in range(0, len(ep_args), gpw)]
                effective_workers = min(num_workers, len(batched_args))
                _log(f"[Iter {current_iteration}] Cross-game batching: {gpw} games/worker, "
                     f"{len(batched_args)} batches across {effective_workers} workers")
                with ctx.Pool(effective_workers, initializer=_init_sp_worker,
                              initargs=sp_initargs) as pool:
                    for multi_result in pool.imap_unordered(
                            _run_multi_episode, batched_args):
                        for result in multi_result:
                            results.append(result)
                            done = len(results)
                            data, p0, p1, turns, elapsed, stats_line = result
                            _log(f"[Iter {current_iteration}] Self-play episode {done}/{num_episodes} done — "
                                 f"score {p0}-{p1}, turns={turns}")
            else:
                _log(f"[Iter {current_iteration}] Spawning pool of {num_workers} workers for {num_episodes} episodes...")
                with ctx.Pool(num_workers, initializer=_init_sp_worker,
                              initargs=sp_initargs) as pool:
                    _log(f"[Iter {current_iteration}] Worker pool ready. Waiting for episodes to complete...")
                    for result in pool.imap_unordered(_run_sp_episode, ep_args):
                        data, p0, p1, turns, elapsed, stats_line = result
                        results.append(result)
                        done = len(results)
                        _log(f"[Iter {current_iteration}] Self-play episode {done}/{num_episodes} done — "
                             f"score {p0}-{p1}, turns={turns}, time={elapsed:.0f}s ({elapsed/60:.1f}min)")

            _log(f"[Iter {current_iteration}] All {num_episodes} episodes finished. Collecting results...")
            for idx, (data, p0, p1, _turns, _elapsed, _stats) in enumerate(results):
                new_samples.extend(data)
                step = current_iteration * num_episodes + idx
                ex.summary_writer.add_scalar(
                    'self_play_episode_player_1_score', p0, global_step=step)
                ex.summary_writer.add_scalar(
                    'self_play_episode_player_2_score', p1, global_step=step)
        else:
            n_random = int(num_episodes * random_opponent_fraction)
            episode_types = [True] * n_random + [False] * (num_episodes - n_random)
            random.shuffle(episode_types)
            for i in tqdm(range(num_episodes), desc="Self play"):
                vs_rand = episode_types[i]
                tag = "vs-random" if vs_rand else "self-play"
                _log(f"[Iter {current_iteration}] Starting {tag} episode {i+1}/{num_episodes}")
                ep_t0 = time.monotonic()
                new_samples.extend(execute_episode(current_iteration * num_episodes + i, vs_random=vs_rand))
                _log(f"[Iter {current_iteration}] Episode {i+1}/{num_episodes} done "
                     f"({time.monotonic() - ep_t0:.1f}s, {len(new_samples)} samples so far)")

        sp_elapsed = time.monotonic() - phase_t0
        _log(f"[Iter {current_iteration}] Self-play complete: {len(new_samples)} new samples in {sp_elapsed:.1f}s ({sp_elapsed/60:.1f}min)")

        # --- Phase 2: Add to replay buffer ---
        phase_t0 = time.monotonic()
        replay_buffer.extend(new_samples)
        _log(f"[Iter {current_iteration}] PHASE 2/5: Replay buffer: {len(replay_buffer)} samples "
             f"(+{len(new_samples)} new, max {replay_buffer_size})")

        # --- Phase 3: Train the model ---
        if len(replay_buffer) < replay_buffer_size:
            _log(f"[Iter {current_iteration}] PHASE 3/5: SKIPPED — replay buffer "
                 f"{len(replay_buffer)}/{replay_buffer_size} (filling)")
        else:
            phase_t0 = time.monotonic()
            _log(f"[Iter {current_iteration}] PHASE 3/5: Saving prev model & training new model")
            agent.save(temp_save_path)
            prev_agent.load(temp_save_path)
            _log(f"[Iter {current_iteration}] Training: {train_steps_per_iter} steps, "
                 f"batch={train_batch_size}, buffer={len(replay_buffer)}")

            agent.train(list(replay_buffer), current_iteration=current_iteration)
            train_elapsed = time.monotonic() - phase_t0
            _log(f"[Iter {current_iteration}] Training complete ({train_elapsed:.1f}s / {train_elapsed/60:.1f}min)")

        # --- Phase 4: Arena (new vs prev) ---
        if len(replay_buffer) < replay_buffer_size:
            _log(f"[Iter {current_iteration}] PHASE 4/5: SKIPPED (buffer filling)")
        elif arena_win_threshold > 0:
            phase_t0 = time.monotonic()
            _log(f"[Iter {current_iteration}] PHASE 4/5: Arena — new model vs previous ({arena_iterations} games)")
            new_sd = {k: v.cpu() for k, v in agent.model.state_dict().items()}
            prev_sd = {k: v.cpu() for k, v in prev_agent.model.state_dict().items()}
            if num_workers > 1:
                new_win, prev_win, prev_new_draw, new_score, prev_score = \
                    play_arena_games_parallel(arena_iterations, new_sd, prev_sd, label="Arena vs prev")
            else:
                new_win, prev_win, prev_new_draw, new_score, prev_score = \
                    play_arena_games_sequential(game, arena_iterations, agent, prev_agent)
            _log(f"[Iter {current_iteration}] Arena result: prev_win={prev_win}, new_win={new_win}, "
                 f"draws={prev_new_draw}, prev_score={prev_score:.1f}, new_score={new_score:.1f} "
                 f"({time.monotonic() - phase_t0:.1f}s)")

            ex.summary_writer.add_scalar('win_rate_against_prev', float(new_win) / arena_iterations, global_step=current_iteration)
            ex.summary_writer.add_scalar('draw_rate_against_prev', float(prev_new_draw) / arena_iterations, global_step=current_iteration)
            ex.summary_writer.add_scalar('prev_score', prev_score, global_step=current_iteration)
            ex.summary_writer.add_scalar('new_score', new_score, global_step=current_iteration)

            if prev_win + new_win == 0 or float(new_win) / (prev_win + new_win) < arena_win_threshold:
                _log(f"[Iter {current_iteration}] REJECTED new model (win rate too low)")
                agent.load(temp_save_path)
            else:
                _log(f"[Iter {current_iteration}] ACCEPTED new model")
        else:
            _log(f"[Iter {current_iteration}] PHASE 4/5: Arena vs previous — SKIPPED (arena_win_threshold=0)")

        # --- Save checkpoint ---
        phase_t0 = time.monotonic()
        _log(f"[Iter {current_iteration}] Saving checkpoint...")
        save()
        _log(f"[Iter {current_iteration}] Checkpoint saved ({time.monotonic() - phase_t0:.1f}s)")

        # --- Phase 5: Arena (current vs random) ---
        phase_t0 = time.monotonic()
        _log(f"[Iter {current_iteration}] PHASE 5/5: Arena — current model vs random ({random_arena_iterations} games)")
        agent_sd = {k: v.cpu() for k, v in agent.model.state_dict().items()}
        if num_workers > 1:
            agent_win, random_win, agent_random_draw, agent_score, random_score = \
                play_arena_games_parallel(random_arena_iterations, agent_sd, None, label="Arena vs random")
        else:
            agent_win, random_win, agent_random_draw, agent_score, random_score = \
                play_arena_games_sequential(game, random_arena_iterations, agent, random_agent)
        _log(f"[Iter {current_iteration}] vs Random result: agent_win={agent_win}, random_win={random_win}, "
             f"draws={agent_random_draw}, agent_score={agent_score:.1f}, random_score={random_score:.1f} "
             f"({time.monotonic() - phase_t0:.1f}s)")

        ex.summary_writer.add_scalar('win_rate_against_random', float(agent_win) / random_arena_iterations, global_step=current_iteration)
        ex.summary_writer.add_scalar('draw_rate_against_random', float(agent_random_draw) / random_arena_iterations, global_step=current_iteration)
        ex.summary_writer.add_scalar('agent_score', agent_score, global_step=current_iteration)
        ex.summary_writer.add_scalar('random_score', random_score, global_step=current_iteration)

        iter_elapsed = time.monotonic() - iter_t0
        _log(f"===== ITERATION {current_iteration} COMPLETE ({iter_elapsed:.1f}s / {iter_elapsed/60:.1f}min) =====")

    _log("Training finished.")
    sys.stdout.flush()
    sys.stderr.flush()


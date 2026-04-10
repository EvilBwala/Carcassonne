from collections import defaultdict
from dataclasses import dataclass
import math
import random
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from utils.utils import argmax_dict, keydefaultdict, normalize_dict
from utils.debug import debug

from games.game_state import GameState


@dataclass
class MCTSConfig:
    num_search: int = 25
    cpuct: float = 1.25
    root_dirichlet_alpha: float = 0.25
    root_exploration_fraction: float = 0.05
    enable_exploration_noise: bool = True
    enable_value_normalization: bool = True
    batch_size: int = 8


class Node(object):
    def __init__(self, parent: Optional['Node']=None, parent_action=None, prior=0) -> None:
        self.parent = parent
        self.parent_action = parent_action
        self.state: GameState = None  # type: ignore
        self.visit_count = 0
        self.value_sum = 0.0
        self.value_estimate = 0.0
        self.expanded = False
        self.prior = prior
        self.children = keydefaultdict(lambda action: Node(self, action, 0))

    def expand(self, state, policy, value):
        self.state = state
        self.value_sum = value
        self.value_estimate = value
        self.expanded = True
        self.children = {
            action: Node(self, action, prob)
            for action, prob in policy.items()
        }


class MCTS(object):
    def __init__(self, model, config: Optional[MCTSConfig]=None,
                 model_batch: Optional[Callable] = None) -> None:
        super().__init__()

        if config is None:
            config = MCTSConfig()

        self.model = model
        self.model_batch = model_batch
        self.config = config
        self.reset()
        self.root = Node()
        self.root_exploration_noise: Optional[Dict[Any, float]] = None

    def reset(self):
        self.current_min = float('inf')
        self.current_max = float('-inf')
        self.all_node_map: Dict[Any, Node] = {}

    def _update_min_max(self, node: Node):
        value = node.value_estimate
        if node.parent and node.parent.state.get_current_player() != node.state.get_current_player():
            value = -value

        self.current_min = min(self.current_min, value)
        self.current_max = max(self.current_max, value)

    def _get_selection_score(self, parent_node: Node, child_node: Node):
        value_estimate = 0.0
        if child_node.expanded:
            value_estimate = child_node.value_estimate

            if parent_node.state.get_current_player() != child_node.state.get_current_player():
                value_estimate = -value_estimate

            if self.config.enable_value_normalization:
                if self.current_max > self.current_min:
                    value_estimate = (value_estimate - self.current_min) / (self.current_max - self.current_min)
                else:
                    value_estimate = 0.0

        prior = child_node.prior
        if parent_node == self.root and self.root_exploration_noise is not None and self.config.enable_exploration_noise:
            prior = prior * (1 - self.config.root_exploration_fraction) + self.root_exploration_noise[child_node.parent_action] * self.config.root_exploration_fraction

        parent_old_count = parent_node.visit_count - 1
        return value_estimate + self.config.cpuct * child_node.prior * math.sqrt(parent_old_count) / (1 + child_node.visit_count)

    def _search(self, node: Node) -> float:
        """Run one iteration of MCTS (sequential). Returns the sample value."""
        node.visit_count += 1

        if debug: print("[debug] mcts.py _search on node, ", node)

        if node.expanded:
            if node.state.is_terminal():
                return node.value_estimate

            scores = {
                action: self._get_selection_score(node, child_node)
                for action, child_node in node.children.items()
            }
            assert len(scores) > 0
            best_action = argmax_dict(scores)
            child_node = node.children[best_action]
            child_sample_return = self._search(child_node)
            if child_node.state.get_current_player() != node.state.get_current_player():
                child_sample_return = -child_sample_return

            node.value_sum += child_sample_return
            node.value_estimate = node.value_sum / node.visit_count

            self._update_min_max(node)
            return child_sample_return

        else:
            if node.parent is None:
                state = node.state
            else:
                state: GameState = node.parent.state.apply_action(node.parent_action)

            sample_return = 0
            if state.is_terminal():
                value = state.get_player_value(state.get_current_player())
                node.expand(state, {}, value)
                sample_return = value
            else:
                all_legal_actions = state.get_legal_actions()
                assert len(all_legal_actions) > 0
                model_output = self.model(state, all_legal_actions)
                pred_policy = model_output['policy']
                pred_value = model_output['value']
                node.expand(state, pred_policy, pred_value)
                sample_return = pred_value

            self._update_min_max(node)

            state_hash_key = state.get_reuse_hash_key()
            if state_hash_key is not None:
                self.all_node_map[state_hash_key] = node

            return sample_return

    # ── Batched leaf evaluation ──────────────────────────────────────────

    def _select_path(self) -> List[Node]:
        """Traverse tree from root to an unexpanded leaf, adding virtual visits."""
        path: List[Node] = []
        node = self.root

        while True:
            node.visit_count += 1
            path.append(node)

            if not node.expanded:
                return path

            if node.state.is_terminal():
                return path

            scores = {
                action: self._get_selection_score(node, child_node)
                for action, child_node in node.children.items()
            }
            assert len(scores) > 0
            best_action = argmax_dict(scores)
            node = node.children[best_action]

    def _backprop_path(self, path: List[Node], leaf_value: float):
        """Backpropagate leaf_value up through path. Visit counts already set."""
        value = leaf_value
        for i in range(len(path) - 2, -1, -1):
            child = path[i + 1]
            parent = path[i]
            if child.state.get_current_player() != parent.state.get_current_player():
                value = -value
            parent.value_sum += value
            parent.value_estimate = parent.value_sum / parent.visit_count
            self._update_min_max(parent)

    def _expand_leaf(self, node: Node, state: GameState, policy: Dict, value: float):
        """Expand a leaf node and register it in the node map."""
        node.expand(state, policy, value)
        self._update_min_max(node)
        state_hash_key = state.get_reuse_hash_key()
        if state_hash_key is not None:
            self.all_node_map[state_hash_key] = node

    # ── Action probability entry point ───────────────────────────────────

    def get_action_probs(self, state: GameState, temperature=1.0, add_exploration_noise=False) -> Dict[Any, float]:
        assert state.get_num_players() == 2, 'Only 2-player zero-sum games are supported.'

        self.current_min = float('inf')
        self.current_max = float('-inf')

        all_legal_actions = state.get_legal_actions()
        assert len(all_legal_actions) > 0

        state_hash_key = state.get_reuse_hash_key()
        if state_hash_key is not None and state_hash_key in self.all_node_map:
            self.root = self.all_node_map[state_hash_key]
        else:
            self.root = Node()
            self.root.state = state

        if add_exploration_noise:
            self.root_exploration_noise = {}
            noise = np.random.dirichlet([self.config.root_dirichlet_alpha] * len(all_legal_actions))
            for a, n in zip(all_legal_actions, noise):
                self.root_exploration_noise[a] = n
        else:
            self.root_exploration_noise = None

        if self.model_batch is not None:
            self._run_batched_search()
        else:
            for i in range(self.config.num_search):
                self._search(self.root)

        all_counts = {
            action: self.root.children[action].visit_count
            for action in all_legal_actions
        }

        if temperature == 0:
            best_count = max(all_counts.values())
            best_actions = [
                action for action, count in all_counts.items()
                if count == best_count
            ]
            best_action = random.choice(best_actions)
            return {best_action: 1.0}
        else:
            counts = {
                action: count ** (1.0 / temperature)
                for action, count in all_counts.items()
            }
            return normalize_dict(counts)  # type: ignore

    def _run_batched_search(self):
        """Multi-round batched MCTS.

        Instead of selecting all paths at once (which limits depth to 1),
        this runs multiple rounds.  Each round selects up to ``batch_size``
        paths, batch-evaluates the leaves, expands them, and back-propagates.
        Subsequent rounds can then traverse through previously expanded nodes,
        building real depth in the search tree.
        """
        sims_remaining = self.config.num_search
        batch_size = self.config.batch_size

        # Ensure root is expanded first (single sequential evaluation).
        if not self.root.expanded:
            self._search(self.root)
            sims_remaining -= 1

        while sims_remaining > 0:
            cur_batch = min(batch_size, sims_remaining)
            sims_remaining -= cur_batch

            pending, resolved = self._select_and_classify(cur_batch)
            if not pending:
                continue

            items = list(pending.values())
            states_actions = [(info[1], info[2]) for info, _ in items]
            results = self.model_batch(states_actions)

            self._expand_from_results(items, results)

    # ── Cross-game batching primitives ────────────────────────────────────

    def prepare_root(self, state: GameState, add_exploration_noise: bool):
        """Set up root for a new search (call once per move, before rounds)."""
        self.current_min = float('inf')
        self.current_max = float('-inf')

        all_legal_actions = state.get_legal_actions()
        assert len(all_legal_actions) > 0

        state_hash_key = state.get_reuse_hash_key()
        if state_hash_key is not None and state_hash_key in self.all_node_map:
            self.root = self.all_node_map[state_hash_key]
        else:
            self.root = Node()
            self.root.state = state

        if add_exploration_noise:
            self.root_exploration_noise = {}
            noise = np.random.dirichlet(
                [self.config.root_dirichlet_alpha] * len(all_legal_actions))
            for a, n in zip(all_legal_actions, noise):
                self.root_exploration_noise[a] = n
        else:
            self.root_exploration_noise = None

        self._sims_remaining = self.config.num_search

        if not self.root.expanded:
            self._search(self.root)
            self._sims_remaining -= 1

    @property
    def search_done(self) -> bool:
        return self._sims_remaining <= 0

    def _select_and_classify(self, cur_batch: int):
        """Select ``cur_batch`` paths and classify leaves.

        Returns (pending_by_leaf, already_resolved_count).
        ``pending_by_leaf`` maps leaf id to ((node, state, legal_actions), [paths]).
        """
        all_paths: List[List[Node]] = []
        for _ in range(cur_batch):
            all_paths.append(self._select_path())

        PendingInfo = Tuple[Node, GameState, List]
        pending_by_leaf: Dict[int, Tuple[PendingInfo, List[List[Node]]]] = {}
        resolved = 0

        for path in all_paths:
            leaf = path[-1]

            if leaf.expanded:
                self._backprop_path(path, leaf.value_estimate)
                resolved += 1
                continue

            leaf_id = id(leaf)
            if leaf_id in pending_by_leaf:
                pending_by_leaf[leaf_id][1].append(path)
                continue

            if leaf.parent is None:
                leaf_state = leaf.state
            else:
                leaf_state = leaf.parent.state.apply_action(leaf.parent_action)

            if leaf_state.is_terminal():
                value = leaf_state.get_player_value(
                    leaf_state.get_current_player())
                self._expand_leaf(leaf, leaf_state, {}, value)
                self._backprop_path(path, value)
                resolved += 1
                continue

            legal_actions = leaf_state.get_legal_actions()
            assert len(legal_actions) > 0
            pending_by_leaf[leaf_id] = (
                (leaf, leaf_state, legal_actions), [path])

        return pending_by_leaf, resolved

    def _expand_from_results(self, items, results):
        """Expand pending leaves with NN results and backpropagate."""
        for ((leaf, leaf_state, _), paths), result in zip(items, results):
            self._expand_leaf(
                leaf, leaf_state, result['policy'], result['value'])
            for path in paths:
                self._backprop_path(path, result['value'])

    def collect_pending_leaves(self) -> List[Tuple[Any, ...]]:
        """Run one round of selection across ``batch_size`` paths.

        Returns a list of (mcts_instance, leaf, state, legal_actions, paths)
        tuples for leaves that need NN evaluation.  Terminal and
        already-expanded leaves are handled internally.

        Call ``deliver_results`` with the NN outputs to complete the round.
        """
        if self._sims_remaining <= 0:
            return []

        cur_batch = min(self.config.batch_size, self._sims_remaining)
        self._sims_remaining -= cur_batch

        pending, _ = self._select_and_classify(cur_batch)

        result = []
        for leaf_id, ((leaf, leaf_state, legal_actions), paths) in pending.items():
            result.append((self, leaf, leaf_state, legal_actions, paths))
        return result

    def deliver_results(self, pending_items, results):
        """Expand leaves using NN results from an external batch call.

        ``pending_items`` is the list returned by ``collect_pending_leaves``.
        ``results`` is a list of {'policy': ..., 'value': ...} in the same order.
        """
        for (_, leaf, leaf_state, _, paths), result in zip(pending_items, results):
            self._expand_leaf(
                leaf, leaf_state, result['policy'], result['value'])
            for path in paths:
                self._backprop_path(path, result['value'])

    def extract_action_probs(self, state: GameState, temperature: float):
        """Extract action probabilities from the completed search tree."""
        all_legal_actions = state.get_legal_actions()
        all_counts = {
            action: self.root.children[action].visit_count
            for action in all_legal_actions
        }

        if temperature == 0:
            best_count = max(all_counts.values())
            best_actions = [
                action for action, count in all_counts.items()
                if count == best_count
            ]
            best_action = random.choice(best_actions)
            return {best_action: 1.0}
        else:
            counts = {
                action: count ** (1.0 / temperature)
                for action, count in all_counts.items()
            }
            return normalize_dict(counts)  # type: ignore

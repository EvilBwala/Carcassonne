from dataclasses import dataclass, field
import os
import random

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from games.game_state import GameState
from utils.context import Context

from .mcts import MCTS, MCTSConfig
from utils.utils import ModelSummarizer, save_model, save_optimizer, load_optimizer, normalize_dict
from .ml_agent import MLAgent
from typing import Any, Dict, List, Optional, Tuple, TypeVar
from utils.debug import debug


@dataclass
class MCTSMLAgentConfig:
    train_steps_per_iter: int = 200
    train_batch_size: int = 256
    mcts_config: MCTSConfig = field(default_factory=MCTSConfig)
    learning_rate: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 1e-4
    gradient_clipping: float = 1.0
    lr_milestones: Tuple[int, ...] = (50, 75)
    lr_gamma: float = 0.1


TState = TypeVar('TState', bound=GameState)


class MCTSMLAgent(MLAgent[TState]):
    def __init__(self, name: str, model: nn.Module, summary_writer, config: Optional[MCTSMLAgentConfig] = None):
        if config is None:
            config = MCTSMLAgentConfig()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.name = name
        self.summary_writer = summary_writer
        self.config = config

        self.model_summarizer = ModelSummarizer(self.model)
        self._build_optimizer()

        self.reset_inference()

    def reset_inference(self):
        self.mcts = self._prepare_mcts()

    def _build_optimizer(self):
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.config.learning_rate,
            momentum=self.config.momentum,
            weight_decay=self.config.weight_decay,
        )

    def _get_loss(self, pred_policy_logit: Tensor, pred_value: Tensor,
                  target_policy: Tensor, target_value: Tensor):
        B = target_value.shape[0]
        log_policy = F.log_softmax(pred_policy_logit, dim=-1)
        policy_loss = -(target_policy * log_policy).sum()
        value_loss = ((target_value.view(B, 1) - pred_value.view(B, 1)) ** 2).sum()
        loss = policy_loss + value_loss
        return policy_loss, value_loss, loss

    def _train_step(self, state_input: Dict[str, Tensor], target_policy: Tensor,
                    target_value: Tensor):
        self.model.train()
        self.optimizer.zero_grad()

        model_output: Any = self.model(state_input)
        pred_policy_logit = model_output['policy_logit']
        pred_value = model_output['value']

        policy_loss, value_loss, loss = self._get_loss(
            pred_policy_logit, pred_value, target_policy, target_value)

        loss.backward()

        if self.config.gradient_clipping > 0:
            nn.utils.clip_grad_value_(self.model.parameters(), self.config.gradient_clipping)

        self.optimizer.step()

        return loss.item(), policy_loss.item(), value_loss.item()

    def get_action_probs(self, state: TState, temperature: float, add_exploration_noise: bool) -> Dict[Any, float]:
        return self.mcts.get_action_probs(state, temperature, add_exploration_noise)

    def _prepare_mcts(self):
        def inference_fn(state, all_legal_actions):
            if debug:
                print("[debug] mcts_ml_agent.py inference_fn performing inference")
            if debug:
                print("[debug] mcts_ml_agent.py inference_fn all_legal_actions = {}".format(
                    all_legal_actions))

            state_input = self._state_to_model_input(state)
            # expand batch dim
            state_input = {k: v.unsqueeze(0).to(self.device)
                           for k, v in state_input.items()}

            if debug:
                for k, v in state_input.items():
                    print("[debug] mcts_ml_agent.py inference_fn state_input[{}].shape = {}".format(
                        k, v.shape))

            self.model.eval()
            with torch.no_grad():
                with self.model_summarizer:
                    model_output: Any = self.model(state_input)

            if debug:
                for k, v in model_output.items():
                    print("[debug] mcts_ml_agent.py inference_fn model_output[{}].shape = {}".format(
                        k, v.shape))

            policy = self._pred_policy_to_action_probs(
                model_output['policy_logit'][0], all_legal_actions)
            value = model_output['value'][0, 0].item()

            if debug:
                print("[debug] mcts_ml_agent.py inference_fn policy = {}".format(policy))
            if debug:
                print("[debug] mcts_ml_agent.py inference_fn value = {}".format(value))

            return {'policy': policy, 'value': value}

        return MCTS(inference_fn, self.config.mcts_config)

    def save(self, path):
        save_model(self.model, path)

    def load(self, path):
        state_dict = torch.load(os.path.join(path, 'model.pth'),
                                map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.reset_inference()

    def train(self, train_data: List[Tuple[TState, Dict[Any, float], float]], current_iteration: int):
        """
        AlphaZero-style training: fresh optimizer + N random mini-batch steps.

        train_data is the replay buffer contents (not modified).
        current_iteration is used for LR milestone scheduling and logging.
        """
        self._build_optimizer()
        self._apply_lr_for_iteration(current_iteration)

        def process_train_data(batch_slice):
            state_input: Dict[str, List[Tensor]] = {}
            target_policy = []
            target_value = []
            for state, policy, value in batch_slice:
                model_input = self._state_to_model_input(state)
                for k, v in model_input.items():
                    if k not in state_input:
                        state_input[k] = []
                    state_input[k].append(v)

                target_policy.append(self._action_probs_to_label(policy))
                target_value.append(float(value))

            batched_input = {k: torch.stack(vs).to(self.device)
                             for k, vs in state_input.items()}
            batched_policy = torch.stack(target_policy).to(self.device)
            batched_value = torch.tensor(target_value, dtype=torch.float32).to(self.device)

            return batched_input, batched_policy, batched_value

        n_steps = self.config.train_steps_per_iter
        bs = self.config.train_batch_size
        buf = list(train_data)
        total_loss = 0.0
        total_ploss = 0.0
        total_vloss = 0.0

        for step in tqdm(range(n_steps), desc='Training', mininterval=1):
            batch = random.choices(buf, k=bs)
            state_input, target_policy, target_value = process_train_data(batch)

            loss_val, policy_loss_val, value_loss_val = self._train_step(
                state_input, target_policy, target_value)

            assert torch.isfinite(torch.tensor(loss_val)), 'Loss is inf or nan'

            total_loss += loss_val
            total_ploss += policy_loss_val
            total_vloss += value_loss_val

        if n_steps > 0:
            global_step = current_iteration
            self.summary_writer.add_scalar(
                f'{self.name}_policy_loss', total_ploss / n_steps, global_step)
            self.summary_writer.add_scalar(
                f'{self.name}_value_loss', total_vloss / n_steps, global_step)
            self.summary_writer.add_scalar(
                f'{self.name}_loss', total_loss / n_steps, global_step)
            self.summary_writer.add_scalar(
                f'{self.name}_lr', self.optimizer.param_groups[0]['lr'], global_step)

    def _apply_lr_for_iteration(self, current_iteration: int):
        """Apply step-decay LR: multiply base LR by gamma for each passed milestone."""
        lr = self.config.learning_rate
        for milestone in sorted(self.config.lr_milestones):
            if current_iteration >= milestone:
                lr *= self.config.lr_gamma
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr

    def _state_to_model_input(self, state: TState) -> Dict[str, Tensor]:
        """
        Return a dict of tensors. Tensors must NOT have a batch dimension.
        Board tensors should be in NCHW format (channels first).
        Subclasses that use a fixed action space must override this method.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement _state_to_model_input"
        )

    def _pred_policy_to_action_probs(self, pred_policy_logit: Tensor,
                                     game_legal_actions) -> Dict[Any, float]:
        """
        Guaranteed to return all legal actions with normalized probabilities.
        pred_policy_logit is a 1-D tensor of shape (ACTION_TOTAL_DIM,).
        Subclasses that use a fixed action space must override this method.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement _pred_policy_to_action_probs"
        )

    def _action_probs_to_label(self, action_probs: Dict[Any, float]) -> Tensor:
        """
        Convert action probs to a dense label tensor of shape (ACTION_TOTAL_DIM,).
        Subclasses that use a fixed action space must override this method.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement _action_probs_to_label"
        )

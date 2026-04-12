from collections import OrderedDict
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .mcts_ml_agent import MCTSMLAgent

try:
    from carcassonne_engine import (
        TileAction, MeepleAction, PassAction, GamePhase, Tile,
    )
    from carcassonne.objects.actions.action import Action
except ImportError:
    from carcassonne.objects.actions.tile_action import TileAction
    from carcassonne.objects.actions.meeple_action import MeepleAction
    from carcassonne.objects.actions.pass_action import PassAction
    from carcassonne.objects.actions.action import Action
    from carcassonne.objects.game_phase import GamePhase
    from carcassonne.objects.tile import Tile

# Always use Python tile data for TileFeatureProvider (needs image paths)
from carcassonne.tile_sets.base_deck import base_tiles
from carcassonne.tile_sets.inns_and_cathedrals_deck import inns_and_cathedrals_tiles

from utils.utils import ModelSummarizer, save_model, save_optimizer, normalize_dict, process_offset_dict
from games.carcassonne_game_state import CarcassonneGameState
from typing import Any, Dict, List, Tuple
from utils.debug import debug


FILE_PATH = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(FILE_PATH, '../carcassonne/resources/images')

TILE_FEATURE_DIM = 576
NUM_MEEPLE_TYPES = 5
NUM_MEEPLE_SIDES = 9
TOTAL_MEEPLES = 7
TOTAL_ABBOTS = 1
TOTAL_BIG_MEEPLES = 1


TILE_FEATURE_OFFSETS: OrderedDict[str, int] = OrderedDict()
TILE_FEATURE_OFFSETS['resnet'] = TILE_FEATURE_DIM
TILE_FEATURE_OFFSETS['just_placed'] = 1
TILE_FEATURE_OFFSETS['is_meeple_friendly'] = 1
TILE_FEATURE_OFFSETS['meeple_type'] = NUM_MEEPLE_TYPES
TILE_FEATURE_OFFSETS['meeple_side'] = NUM_MEEPLE_SIDES

TILE_FEATURE_TOTAL_DIM = process_offset_dict(TILE_FEATURE_OFFSETS)

GAME_FEATURE_OFFSETS: OrderedDict[str, int] = OrderedDict()
GAME_FEATURE_OFFSETS['next_tile_resnet_0'] = TILE_FEATURE_DIM
GAME_FEATURE_OFFSETS['next_tile_resnet_1'] = TILE_FEATURE_DIM
GAME_FEATURE_OFFSETS['next_tile_resnet_2'] = TILE_FEATURE_DIM
GAME_FEATURE_OFFSETS['next_tile_resnet_3'] = TILE_FEATURE_DIM
GAME_FEATURE_OFFSETS['player_meeples'] = TOTAL_MEEPLES
GAME_FEATURE_OFFSETS['player_abbots'] = TOTAL_ABBOTS
GAME_FEATURE_OFFSETS['player_big_meeples'] = TOTAL_BIG_MEEPLES
GAME_FEATURE_OFFSETS['enemy_meeples'] = TOTAL_MEEPLES
GAME_FEATURE_OFFSETS['enemy_abbots'] = TOTAL_ABBOTS
GAME_FEATURE_OFFSETS['enemy_big_meeples'] = TOTAL_BIG_MEEPLES
GAME_FEATURE_OFFSETS['is_meeples_phase'] = 1
GAME_FEATURE_OFFSETS['current_value'] = 1

GAME_FEATURE_TOTAL_DIM = process_offset_dict(GAME_FEATURE_OFFSETS)

BOARD_SIZE = 10

ACTION_OFFSETS: OrderedDict[str, int] = OrderedDict()
TILE_ACTION_DIM = BOARD_SIZE * BOARD_SIZE * 4
ACTION_OFFSETS['place_tile'] = TILE_ACTION_DIM
ACTION_OFFSETS['place_meeple'] = NUM_MEEPLE_SIDES * NUM_MEEPLE_TYPES
ACTION_OFFSETS['remove_abbot'] = 1
ACTION_OFFSETS['pass'] = 1

ACTION_TOTAL_DIM = process_offset_dict(ACTION_OFFSETS)
OTHER_ACTION_DIM = ACTION_TOTAL_DIM - TILE_ACTION_DIM


def action_to_idx(action: Action):
    if isinstance(action, PassAction):
        return ACTION_OFFSETS['pass']

    if isinstance(action, TileAction):
        return ACTION_OFFSETS['place_tile'] + action.coordinate.row * BOARD_SIZE * 4 + action.coordinate.column * 4 + action.tile_rotations

    if isinstance(action, MeepleAction):
        if action.remove:
            return ACTION_OFFSETS['remove_abbot']

        return ACTION_OFFSETS['place_meeple'] + action.coordinate_with_side.side * NUM_MEEPLE_TYPES + action.meeple_type

    raise Exception('Unknown action type: ' + str(action))


class TileFeatureProvider(object):
    def __init__(self):
        super().__init__()
        self.cache = {}
        # prepopulate using known tiles
        for tile in base_tiles.values():
            self.get_tile_feature(tile)

        for tile in inns_and_cathedrals_tiles.values():
            self.get_tile_feature(tile)

    def get_tile_feature(self, tile: Tile) -> np.ndarray:
        id = tile.description
        if id not in self.cache:
            features = np.load(os.path.join(
                IMAGE_DIR, tile.image + '.feat.npy'))
            assert len(features) == 4
            assert len(features[0]) == TILE_FEATURE_DIM
            self.cache[id] = features

        return self.cache[id][tile.turns]


tile_feature_provider = TileFeatureProvider()

LATENT_DIM = 128
BACKBONE_LATENT_DIM = 128
DROPOUT = 0.2
GAME_FEATURE_LATENT_DIM = 256
POLICY_TILE_HEAD_LATENT_DIM = 128
VALUE_HEAD_INIT_STDDEV = 0.01

BOOL_FEATURE_SCALE = 20.0


class DownConvLayer(nn.Module):
    """Single downsampling conv block: Conv2d → BatchNorm2d → LeakyReLU."""

    def __init__(self, in_channels: int, out_channels: int = LATENT_DIM):
        super().__init__()
        self.sequence = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.sequence(x)


class UpConvLayer(nn.Module):
    """Single upsampling deconv block: ConvTranspose2d → BatchNorm2d → LeakyReLU."""

    def __init__(self, in_channels: int, out_channels: int = POLICY_TILE_HEAD_LATENT_DIM):
        super().__init__()
        self.sequence = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.sequence(x)


class BoardEncoder(nn.Module):
    """
    Encodes the 10×10 board from (B, TILE_FEATURE_TOTAL_DIM, 10, 10)
    down to (B, LATENT_DIM, 2, 2) through 4 DownConvLayers.

    Also returns the 4 intermediate feature maps for skip connections
    in the policy head decoder.
    """

    def __init__(self):
        super().__init__()
        self.layer0 = DownConvLayer(TILE_FEATURE_TOTAL_DIM, LATENT_DIM)
        self.layer1 = DownConvLayer(LATENT_DIM, LATENT_DIM)
        self.layer2 = DownConvLayer(LATENT_DIM, LATENT_DIM)
        self.layer3 = DownConvLayer(LATENT_DIM, LATENT_DIM)

    def forward(self, x: Tensor):
        intermediate_features = []
        for layer in (self.layer0, self.layer1, self.layer2, self.layer3):
            intermediate_features.append(x)
            x = layer(x)
        # x: (B, LATENT_DIM, 2, 2)
        # intermediate_features[i] = input before layer i
        return x, intermediate_features


class GameEncoder(nn.Module):
    """Encodes the flat game feature vector (B, GAME_FEATURE_TOTAL_DIM) → (B, GAME_FEATURE_LATENT_DIM)."""

    def __init__(self):
        super().__init__()
        self.sequence = nn.Sequential(
            nn.Linear(GAME_FEATURE_TOTAL_DIM, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Linear(512, GAME_FEATURE_LATENT_DIM),
            nn.BatchNorm1d(GAME_FEATURE_LATENT_DIM),
            nn.LeakyReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.sequence(x)


class Backbone(nn.Module):
    """
    Shared backbone that fuses board + game features.
    Input: (B, LATENT_DIM + GAME_FEATURE_LATENT_DIM, 2, 2)
    Output: (B, BACKBONE_LATENT_DIM, 2, 2)
    """

    def __init__(self):
        super().__init__()
        in_channels = LATENT_DIM + GAME_FEATURE_LATENT_DIM
        self.sequence = nn.Sequential(
            nn.Conv2d(in_channels, BACKBONE_LATENT_DIM, kernel_size=3, padding='same'),
            nn.BatchNorm2d(BACKBONE_LATENT_DIM),
            nn.LeakyReLU(),
            nn.Dropout2d(DROPOUT),
            nn.Conv2d(BACKBONE_LATENT_DIM, BACKBONE_LATENT_DIM, kernel_size=1),
            nn.BatchNorm2d(BACKBONE_LATENT_DIM),
            nn.LeakyReLU(),
            nn.Dropout2d(DROPOUT),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.sequence(x)


class PolicyHead(nn.Module):
    """
    Policy head with a U-Net style decoder for tile placement logits
    and a separate MLP branch for meeple/pass logits.

    Tile decoder upsampling path (with skip connections from BoardEncoder):
      x (B,128,2,2)  →upconv1→  (B,128,4,4)
      cat skip[-1]   →upconv2→  (B,128,6,6)   [input: 256 ch]
      cat skip[-2]   →upconv3→  (B,128,8,8)   [input: 256 ch]
      cat skip[-3]   →final →   (B,4,10,10)   [input: 256 ch]
    """

    def __init__(self):
        super().__init__()
        # Tile action decoder
        self.upconv1 = UpConvLayer(BACKBONE_LATENT_DIM, POLICY_TILE_HEAD_LATENT_DIM)
        self.upconv2 = UpConvLayer(POLICY_TILE_HEAD_LATENT_DIM + LATENT_DIM, POLICY_TILE_HEAD_LATENT_DIM)
        self.upconv3 = UpConvLayer(POLICY_TILE_HEAD_LATENT_DIM + LATENT_DIM, POLICY_TILE_HEAD_LATENT_DIM)
        self.final_conv = nn.ConvTranspose2d(POLICY_TILE_HEAD_LATENT_DIM + LATENT_DIM, 4, kernel_size=3)

        # Meeple / pass action branch
        self.other_action_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(BACKBONE_LATENT_DIM * 2 * 2, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Linear(512, OTHER_ACTION_DIM),
        )

    def forward(self, x: Tensor, intermediate_features: List[Tensor]) -> Tensor:
        B = x.shape[0]
        other_action = self.other_action_head(x)

        # Decoder path with skip connections
        x = self.upconv1(x)                                                # (B,128,4,4)
        x = torch.cat([x, intermediate_features[-1]], dim=1)               # (B,256,4,4)
        x = self.upconv2(x)                                                # (B,128,6,6)
        x = torch.cat([x, intermediate_features[-2]], dim=1)               # (B,256,6,6)
        x = self.upconv3(x)                                                # (B,128,8,8)
        x = torch.cat([x, intermediate_features[-3]], dim=1)               # (B,256,8,8)
        x = self.final_conv(x)                                             # (B,4,10,10)

        tile_action = x.view(B, -1)                                        # (B, 400)
        return torch.cat([tile_action, other_action], dim=-1)              # (B, 447)


class ValueHead(nn.Module):
    """
    Value head: flattens backbone output and produces a scalar in [-1, 1].
    Input: (B, BACKBONE_LATENT_DIM, 2, 2)
    Output: (B, 1)
    """

    def __init__(self):
        super().__init__()
        self.sequence = nn.Sequential(
            nn.Flatten(),
            nn.Linear(BACKBONE_LATENT_DIM * 2 * 2, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Linear(512, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return torch.tanh(self.sequence(x))


class CarcassonneModel(nn.Module):
    """
    Full AlphaZero model for Carcassonne.

    Inputs:
      x['board']: (B, TILE_FEATURE_TOTAL_DIM, 10, 10)  - NCHW board tensor
      x['game']:  (B, GAME_FEATURE_TOTAL_DIM)           - flat game feature vector

    Outputs:
      {'policy_logit': (B, ACTION_TOTAL_DIM), 'value': (B, 1)}
    """

    def __init__(self):
        super().__init__()
        self.board_encoder = BoardEncoder()
        self.game_encoder = GameEncoder()
        self.backbone = Backbone()
        self.policy_head = PolicyHead()
        self.value_head = ValueHead()

    def forward(self, x: Dict[str, Tensor]) -> Dict[str, Tensor]:
        board = x['board']   # (B, TILE_FEATURE_TOTAL_DIM, 10, 10)
        game = x['game']     # (B, GAME_FEATURE_TOTAL_DIM)
        B = board.shape[0]

        board_feature, intermediate_features = self.board_encoder(board)
        # board_feature: (B, 128, 2, 2)

        game_feature = self.game_encoder(game)
        # game_feature: (B, 256) → tile spatially to (B, 256, 2, 2)
        game_feature = game_feature.view(B, GAME_FEATURE_LATENT_DIM, 1, 1).expand(-1, -1, 2, 2)

        feature = torch.cat([board_feature, game_feature], dim=1)  # (B, 384, 2, 2)
        feature = self.backbone(feature)                            # (B, 128, 2, 2)

        policy_logit = self.policy_head(feature, intermediate_features)
        value = self.value_head(feature)

        return {'policy_logit': policy_logit, 'value': value}


class CarcassonneAgent(MCTSMLAgent[CarcassonneGameState]):
    def __init__(self, name: str, summary_writer):
        super().__init__(name, CarcassonneModel(), summary_writer)

    def _state_to_model_input(self, state: CarcassonneGameState) -> Dict[str, Tensor]:
        lib_state = state.lib_state
        player = state.get_current_player()

        num_rows = len(lib_state.board)
        num_columns = len(lib_state.board[0])
        assert num_rows == BOARD_SIZE and num_columns == BOARD_SIZE

        # Build (H, W, C) array then transpose to (C, H, W) for PyTorch
        board_input = np.zeros((num_rows, num_columns, TILE_FEATURE_TOTAL_DIM), dtype=np.float32)

        for row in range(num_rows):
            for column in range(num_columns):
                tile = lib_state.board[row][column]
                if tile is None:
                    continue

                offset = TILE_FEATURE_OFFSETS['resnet']
                feature = tile_feature_provider.get_tile_feature(tile)
                if debug.tile_feature:
                    print("[debug] carcassonne_agent.py tile({},{}) = {}".format(
                        row, column, feature[:10]))
                board_input[row, column, offset:offset + TILE_FEATURE_DIM] = feature

        last_tile_action = lib_state.last_tile_action
        if last_tile_action is not None:
            row = last_tile_action.coordinate.row
            column = last_tile_action.coordinate.column
            offset = TILE_FEATURE_OFFSETS['just_placed']
            board_input[row, column, offset] = BOOL_FEATURE_SCALE

        for meeple_player, meeples in enumerate(lib_state.placed_meeples):
            is_friendly = int(meeple_player == player) * BOOL_FEATURE_SCALE
            for meeple in meeples:
                row = meeple.coordinate_with_side.coordinate.row
                column = meeple.coordinate_with_side.coordinate.column
                offset = TILE_FEATURE_OFFSETS['is_meeple_friendly']
                board_input[row, column, offset] = is_friendly
                offset = TILE_FEATURE_OFFSETS['meeple_type']
                board_input[row, column, offset + meeple.meeple_type] = BOOL_FEATURE_SCALE
                offset = TILE_FEATURE_OFFSETS['meeple_side']
                board_input[row, column, offset + meeple.coordinate_with_side.side] = BOOL_FEATURE_SCALE

        game_input = np.zeros((GAME_FEATURE_TOTAL_DIM,), dtype=np.float32)

        if lib_state.next_tile is not None:
            if debug.tile_feature:
                print("[debug] carcassonne_agent.py next_tile = {}".format(
                    lib_state.next_tile.description))

            for rotation in range(4):
                offset = GAME_FEATURE_OFFSETS[f'next_tile_resnet_{rotation}']
                feature = tile_feature_provider.get_tile_feature(
                    lib_state.next_tile.turn(rotation))
                game_input[offset:offset + TILE_FEATURE_DIM] = feature
                if debug.tile_feature:
                    print("[debug] carcassonne_agent.py next_tile_feature({}) = {}".format(
                        rotation, feature[:10]))

        offset = GAME_FEATURE_OFFSETS['player_meeples']
        for i in range(lib_state.meeples[player]):
            game_input[offset + i] = BOOL_FEATURE_SCALE

        offset = GAME_FEATURE_OFFSETS['player_abbots']
        for i in range(lib_state.abbots[player]):
            game_input[offset + i] = BOOL_FEATURE_SCALE

        offset = GAME_FEATURE_OFFSETS['player_big_meeples']
        for i in range(lib_state.big_meeples[player]):
            game_input[offset + i] = BOOL_FEATURE_SCALE

        enemy = 1 - player
        offset = GAME_FEATURE_OFFSETS['enemy_meeples']
        for i in range(lib_state.meeples[enemy]):
            game_input[offset + i] = BOOL_FEATURE_SCALE

        offset = GAME_FEATURE_OFFSETS['enemy_abbots']
        for i in range(lib_state.abbots[enemy]):
            game_input[offset + i] = BOOL_FEATURE_SCALE

        offset = GAME_FEATURE_OFFSETS['enemy_big_meeples']
        for i in range(lib_state.big_meeples[enemy]):
            game_input[offset + i] = BOOL_FEATURE_SCALE

        offset = GAME_FEATURE_OFFSETS['is_meeples_phase']
        game_input[offset] = int(lib_state.phase == GamePhase.MEEPLES) * BOOL_FEATURE_SCALE

        offset = GAME_FEATURE_OFFSETS['current_value']
        game_input[offset] = float(state.get_player_value(player))

        # Transpose board from (H, W, C) to (C, H, W) for PyTorch NCHW convention
        board_tensor = torch.from_numpy(board_input.transpose(2, 0, 1))
        game_tensor = torch.from_numpy(game_input)

        return {'board': board_tensor, 'game': game_tensor}

    def _pred_policy_to_action_probs(self, pred_policy_logit: Tensor,
                                     game_legal_actions) -> Dict[Any, float]:
        """
        Applies softmax, masks to legal action indices, re-normalizes.
        Falls back to uniform policy if all legal-action probabilities are zero.
        """
        if debug:
            print("[debug] carcassonne_agent.py _pred_policy_to_action_probs")
        assert len(game_legal_actions) > 0

        pred_policy = torch.softmax(pred_policy_logit, dim=0)

        if debug:
            for action in game_legal_actions:
                if action_to_idx(action) >= 442:
                    print("[debug] carcassonne_agent.py action = {}".format(action))

        game_legal_action_indices = [action_to_idx(action) for action in game_legal_actions]
        pred_policy_numpy = pred_policy.cpu().detach().numpy()
        game_legal_action_probs = pred_policy_numpy[game_legal_action_indices]

        policy = {
            action: float(prob)
            for action, prob in zip(game_legal_actions, game_legal_action_probs)
        }
        # re-normalize to ensure probabilities sum to 1
        policy = normalize_dict(policy)
        if policy is None:
            # fallback: model assigned zero prob to all legal actions
            policy = {
                action: 1.0 / len(game_legal_actions)
                for action in game_legal_actions
            }

        if debug:
            print("[debug] carcassonne_agent.py policy = {}".format(policy))

        return policy

    def _action_probs_to_label(self, action_probs: Dict[Any, float]) -> Tensor:
        """Convert {action: prob} dict to a dense label tensor of shape (ACTION_TOTAL_DIM,)."""
        label = np.zeros((ACTION_TOTAL_DIM,), dtype=np.float32)
        for action, prob in action_probs.items():
            label[action_to_idx(action)] = prob

        return torch.from_numpy(label)

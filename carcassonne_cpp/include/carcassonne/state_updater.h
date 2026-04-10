#pragma once
#include "actions.h"
#include "game_state.h"
#include "meeple_position.h"
#include "points_collector.h"
#include "types.h"

namespace carcassonne {

class StateUpdater {
public:
    static void next_player(CarcassonneGameState& state) {
        state.phase = GamePhase::TILES;
        state.current_player = (state.current_player + 1) % state.players;
    }

    static void play_tile(CarcassonneGameState& state, const TileAction& ta) {
        state.board[{ta.coordinate.row, ta.coordinate.column}] = ta.tile;
        state.phase = GamePhase::MEEPLES;
        state.last_tile_action = ta;
    }

    static void play_meeple(CarcassonneGameState& state, const MeepleAction& ma) {
        if (!ma.remove) {
            state.placed_meeples[state.current_player].push_back(
                MeeplePosition(ma.meeple_type, ma.coordinate_with_side));
        } else {
            auto& pm = state.placed_meeples[state.current_player];
            MeeplePosition target(ma.meeple_type, ma.coordinate_with_side);
            auto it = std::find(pm.begin(), pm.end(), target);
            if (it != pm.end()) pm.erase(it);
        }

        if (ma.meeple_type == MeepleType::NORMAL || ma.meeple_type == MeepleType::FARMER) {
            state.meeples[state.current_player] += ma.remove ? 1 : -1;
        } else if (ma.meeple_type == MeepleType::ABBOT) {
            if (ma.remove) {
                int points = PointsCollector::chapel_or_flowers_points(
                    state, ma.coordinate_with_side.coordinate);
                state.scores[state.current_player] += points;
            }
            state.abbots[state.current_player] += ma.remove ? 1 : -1;
        } else if (ma.meeple_type == MeepleType::BIG || ma.meeple_type == MeepleType::BIG_FARMER) {
            state.big_meeples[state.current_player] += ma.remove ? 1 : -1;
        }
    }

    static void draw_tile(CarcassonneGameState& state) {
        if (state.deck.empty()) {
            state.next_tile = std::nullopt;
        } else {
            state.next_tile = state.deck.front();
            state.deck.erase(state.deck.begin());
        }
    }

    static void remove_meeples_and_update_score(CarcassonneGameState& state) {
        if (state.last_tile_action.has_value()) {
            PointsCollector::remove_meeples_and_collect_points(
                state, state.last_tile_action->coordinate);
        }
    }

    static CarcassonneGameState apply_action(const CarcassonneGameState& state,
                                              const Action& action) {
        CarcassonneGameState ns = state.simple_copy();

        if (action.type() == ActionType::TILE) {
            ns.board = std::unordered_map<std::pair<int,int>, Tile, PairHash>(ns.board);
            const auto& ta = static_cast<const TileAction&>(action);
            play_tile(ns, ta);
            ns.phase = GamePhase::MEEPLES;
        } else if (action.type() == ActionType::MEEPLE) {
            const auto& ma = static_cast<const MeepleAction&>(action);
            play_meeple(ns, ma);
        } else { // PassAction
            if (state.phase == GamePhase::TILES) {
                ns.last_tile_action = std::nullopt;
                ns.phase = GamePhase::MEEPLES;
            }
        }

        if (state.phase == GamePhase::MEEPLES) {
            remove_meeples_and_update_score(ns);
            draw_tile(ns);
            next_player(ns);
        }

        if (ns.is_terminated()) {
            PointsCollector::count_final_scores(ns);
        }

        return ns;
    }
};

} // namespace carcassonne

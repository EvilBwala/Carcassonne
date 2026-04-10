#pragma once
#include <algorithm>
#include <optional>
#include <vector>
#include "coordinate.h"
#include "game_state.h"
#include "meeple_position.h"
#include "types.h"

namespace carcassonne {

class MeepleUtil {
public:
    static std::optional<int> position_contains_meeple(
            const CarcassonneGameState& state, const CoordinateWithSide& cws) {
        for (int player = 0; player < state.players; ++player) {
            for (auto& mp : state.placed_meeples[player]) {
                if (cws == mp.coordinate_with_side) return player;
            }
        }
        return std::nullopt;
    }

    static void remove_meeples(CarcassonneGameState& state,
                                const std::vector<std::vector<MeeplePosition>>& meeples) {
        for (int player = 0; player < (int)meeples.size(); ++player) {
            for (auto& mp : meeples[player]) {
                remove_meeple(state, mp, player);
            }
        }
    }

    static void remove_meeple(CarcassonneGameState& state, const MeeplePosition& mp, int player) {
        auto& pm = state.placed_meeples[player];
        auto it = std::find(pm.begin(), pm.end(), mp);
        if (it != pm.end()) pm.erase(it);

        if (mp.meeple_type == MeepleType::NORMAL || mp.meeple_type == MeepleType::FARMER) {
            state.meeples[player] += 1;
        } else if (mp.meeple_type == MeepleType::ABBOT) {
            state.abbots[player] += 1;
        } else if (mp.meeple_type == MeepleType::BIG || mp.meeple_type == MeepleType::BIG_FARMER) {
            state.big_meeples[player] += 1;
        }
    }
};

} // namespace carcassonne

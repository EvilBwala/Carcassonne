#pragma once
#include <vector>
#include "actions.h"
#include "game_state.h"
#include "possible_move_finder.h"
#include "structures.h"
#include "tile_position_finder.h"
#include "types.h"

namespace carcassonne {

class ActionUtil {
public:
    static std::vector<ActionPtr> get_possible_actions(const CarcassonneGameState& state) {
        std::vector<ActionPtr> actions;

        if (state.phase == GamePhase::TILES) {
            if (!state.next_tile.has_value()) return actions;

            auto positions = TilePositionFinder::possible_playing_positions(
                state, *state.next_tile);

            if (positions.empty()) {
                actions.push_back(make_pass_action());
            } else {
                for (auto& pp : positions) {
                    actions.push_back(make_tile_action(
                        state.next_tile->turn(pp.turns),
                        pp.coordinate,
                        pp.turns
                    ));
                }
            }
        } else { // MEEPLES phase
            auto meeple_actions = PossibleMoveFinder::possible_meeple_actions(state);
            for (auto& a : meeple_actions) actions.push_back(std::move(a));
            actions.push_back(make_pass_action());
        }
        return actions;
    }
};

} // namespace carcassonne

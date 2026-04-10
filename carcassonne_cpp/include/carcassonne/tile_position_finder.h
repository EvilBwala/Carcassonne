#pragma once
#include <set>
#include <vector>
#include "coordinate.h"
#include "game_state.h"
#include "structures.h"
#include "tile.h"
#include "tile_fitter.h"

namespace carcassonne {

class TilePositionFinder {
public:
    static std::vector<PlayingPosition> possible_playing_positions(
            const CarcassonneGameState& state, const Tile& tile_to_play) {
        if (state.empty_board()) {
            return {PlayingPosition(state.starting_position, 0)};
        }

        std::set<std::pair<int,int>> candidate_slots;
        for (auto& [pos, _] : state.board) {
            int r = pos.first, c = pos.second;
            for (auto [dr, dc] : std::initializer_list<std::pair<int,int>>{{-1,0},{1,0},{0,-1},{0,1}}) {
                int nr = r + dr, nc = c + dc;
                if (!state.get_tile(nr, nc)) {
                    candidate_slots.insert({nr, nc});
                }
            }
        }

        std::vector<PlayingPosition> result;
        for (auto& [row, col] : candidate_slots) {
            const Tile* top = state.get_tile(row - 1, col);
            const Tile* bottom = state.get_tile(row + 1, col);
            const Tile* left = state.get_tile(row, col - 1);
            const Tile* right = state.get_tile(row, col + 1);

            for (int turns = 0; turns < 4; ++turns) {
                Tile rotated = tile_to_play.turn(turns);
                if (TileFitter::fits(rotated, top, right, bottom, left, &state)) {
                    result.push_back(PlayingPosition(Coordinate(row, col), turns));
                }
            }
        }
        return result;
    }
};

} // namespace carcassonne

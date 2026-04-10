#pragma once
#include <optional>
#include <set>
#include <vector>
#include "connection.h"
#include "coordinate.h"
#include "game_state.h"
#include "meeple_position.h"
#include "side_modification_util.h"
#include "structures.h"
#include "tile.h"
#include "types.h"

namespace carcassonne {

class FarmUtil {
public:
    static Farm find_farm_by_coordinate(const CarcassonneGameState& state,
                                         const CoordinateWithSide& position) {
        const Tile* tile = state.get_tile(position.coordinate.row, position.coordinate.column);
        if (!tile) return Farm();

        for (auto& fc : tile->farms) {
            for (int fp : fc.farmer_positions) {
                if (fp == position.side) {
                    return find_farm(state, FarmerConnectionWithCoordinate(fc, position.coordinate));
                }
            }
        }
        return Farm();
    }

    static Farm find_farm(const CarcassonneGameState& state,
                           const FarmerConnectionWithCoordinate& fcwc) {
        std::set<FarmerConnectionWithCoordinate> farmer_connections = {fcwc};

        std::set<CoordinateWithFarmerSide> open_sides;
        for (int tc : fcwc.farmer_connection.tile_connections) {
            open_sides.insert(CoordinateWithFarmerSide(fcwc.coordinate, tc));
        }

        std::set<CoordinateWithFarmerSide> to_explore;
        for (auto& os : open_sides) {
            to_explore.insert(opposite_edge(os));
        }

        std::set<CoordinateWithFarmerSide> to_ignore = open_sides;
        to_ignore.insert(to_explore.begin(), to_explore.end());

        while (!to_explore.empty()) {
            auto it = to_explore.begin();
            CoordinateWithFarmerSide open_edge = *it;
            to_explore.erase(it);
            to_ignore.insert(open_edge);

            auto new_fc = farm_for_position(state, open_edge);
            if (new_fc.has_value()) {
                farmer_connections.insert(*new_fc);

                std::set<CoordinateWithFarmerSide> new_open_sides;
                for (int tc : new_fc->farmer_connection.tile_connections) {
                    new_open_sides.insert(CoordinateWithFarmerSide(new_fc->coordinate, tc));
                }

                to_ignore.insert(new_open_sides.begin(), new_open_sides.end());

                for (auto& nos : new_open_sides) {
                    CoordinateWithFarmerSide ne = opposite_edge(nos);
                    if (to_ignore.find(ne) == to_ignore.end()) {
                        to_explore.insert(ne);
                        to_ignore.insert(ne);
                    }
                }
            }
        }

        return Farm(farmer_connections);
    }

    static CoordinateWithFarmerSide opposite_edge(const CoordinateWithFarmerSide& cwfs) {
        int side = FarmerSide::get_side(cwfs.farmer_side);
        int opp_fs = SideModificationUtil::opposite_farmer_side(cwfs.farmer_side);

        if (side == Side::TOP)
            return CoordinateWithFarmerSide(
                Coordinate(cwfs.coordinate.row - 1, cwfs.coordinate.column), opp_fs);
        if (side == Side::RIGHT)
            return CoordinateWithFarmerSide(
                Coordinate(cwfs.coordinate.row, cwfs.coordinate.column + 1), opp_fs);
        if (side == Side::BOTTOM)
            return CoordinateWithFarmerSide(
                Coordinate(cwfs.coordinate.row + 1, cwfs.coordinate.column), opp_fs);
        if (side == Side::LEFT)
            return CoordinateWithFarmerSide(
                Coordinate(cwfs.coordinate.row, cwfs.coordinate.column - 1), opp_fs);
        return cwfs;
    }

    static std::optional<FarmerConnectionWithCoordinate> farm_for_position(
            const CarcassonneGameState& state, const CoordinateWithFarmerSide& cwfs) {
        const Tile* tile = state.get_tile(cwfs.coordinate.row, cwfs.coordinate.column);
        if (!tile) return std::nullopt;

        for (auto& fc : tile->farms) {
            for (int tc : fc.tile_connections) {
                if (tc == cwfs.farmer_side) {
                    return FarmerConnectionWithCoordinate(fc, cwfs.coordinate);
                }
            }
        }
        return std::nullopt;
    }

    static bool has_meeples(const CarcassonneGameState& state, const Farm& farm) {
        auto meeples = find_meeples(state, farm);
        for (auto& player_meeples : meeples) {
            if (!player_meeples.empty()) return true;
        }
        return false;
    }

    static std::vector<std::vector<MeeplePosition>> find_meeples(
            const CarcassonneGameState& state, const Farm& farm) {
        std::vector<std::vector<MeeplePosition>> meeples(state.players);

        for (auto& fcwc : farm.farmer_connections_with_coordinate) {
            if (fcwc.farmer_connection.farmer_positions.empty()) continue;
            int fp = fcwc.farmer_connection.farmer_positions[0];
            CoordinateWithSide farmer_pos(fcwc.coordinate, fp);
            for (int player = 0; player < state.players; ++player) {
                for (auto& mp : state.placed_meeples[player]) {
                    if (farmer_pos == mp.coordinate_with_side) {
                        meeples[player].push_back(mp);
                    }
                }
            }
        }
        return meeples;
    }
};

} // namespace carcassonne

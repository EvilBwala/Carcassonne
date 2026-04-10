#pragma once
#include <algorithm>
#include <vector>
#include "actions.h"
#include "city_util.h"
#include "connection.h"
#include "coordinate.h"
#include "farm_util.h"
#include "game_state.h"
#include "road_util.h"
#include "structures.h"
#include "tile.h"
#include "types.h"

namespace carcassonne {

class PossibleMoveFinder {
public:
    static std::vector<ActionPtr> possible_meeple_actions(const CarcassonneGameState& state) {
        int current_player = state.current_player;
        if (!state.last_tile_action.has_value()) return {};
        const TileAction& lta = *state.last_tile_action;
        const Tile& last_tile = lta.tile;
        const Coordinate& last_pos = lta.coordinate;

        std::vector<ActionPtr> actions;

        auto meeple_positions = possible_meeple_positions(state);

        bool has_farmers = std::find(state.supplementary_rules.begin(),
                                      state.supplementary_rules.end(),
                                      SupplementaryRule::FARMERS) != state.supplementary_rules.end();
        std::vector<CoordinateWithSide> farmer_positions;
        if (has_farmers) {
            farmer_positions = possible_farmer_positions(state);
        }

        if (state.meeples[current_player] > 0) {
            for (auto& pos : meeple_positions) {
                actions.push_back(make_meeple_action(MeepleType::NORMAL, pos));
            }
            for (auto& pos : farmer_positions) {
                actions.push_back(make_meeple_action(MeepleType::FARMER, pos));
            }
        }

        if (state.big_meeples[current_player] > 0) {
            for (auto& pos : meeple_positions) {
                actions.push_back(make_meeple_action(MeepleType::BIG, pos));
            }
            for (auto& pos : farmer_positions) {
                actions.push_back(make_meeple_action(MeepleType::BIG_FARMER, pos));
            }
        }

        if (state.abbots[current_player] > 0) {
            if (last_tile.chapel || last_tile.flowers) {
                actions.push_back(make_meeple_action(
                    MeepleType::ABBOT, CoordinateWithSide(last_pos, Side::CENTER)));
            }
        }

        for (auto& placed : state.placed_meeples[current_player]) {
            if (placed.meeple_type == MeepleType::ABBOT) {
                actions.push_back(make_meeple_action(
                    MeepleType::ABBOT, placed.coordinate_with_side, true));
            }
        }

        return actions;
    }

private:
    static std::vector<CoordinateWithSide> possible_meeple_positions(
            const CarcassonneGameState& state) {
        std::vector<CoordinateWithSide> positions;
        const TileAction& lta = *state.last_tile_action;
        const Tile& tile = lta.tile;
        const Coordinate& pos = lta.coordinate;

        if (tile.chapel) {
            positions.push_back(CoordinateWithSide(pos, Side::CENTER));
        }

        bool flowers_ok = std::find(state.supplementary_rules.begin(),
                                     state.supplementary_rules.end(),
                                     SupplementaryRule::NORMAL_MEEPLES_CAN_USE_FLOWERS)
                          != state.supplementary_rules.end();
        if (tile.flowers && flowers_ok) {
            positions.push_back(CoordinateWithSide(pos, Side::CENTER));
        }

        for (int side : {Side::TOP, Side::RIGHT, Side::BOTTOM, Side::LEFT}) {
            if (tile.get_type(side) == TerrainType::CITY) {
                City city = CityUtil::find_city(state, CoordinateWithSide(pos, side));
                if (!CityUtil::city_contains_meeples(state, city)) {
                    positions.push_back(CoordinateWithSide(pos, side));
                }
            }
            if (tile.get_type(side) == TerrainType::ROAD) {
                Road road = RoadUtil::find_road(state, CoordinateWithSide(pos, side));
                if (!RoadUtil::road_contains_meeples(state, road)) {
                    positions.push_back(CoordinateWithSide(pos, side));
                }
            }
        }
        return positions;
    }

    static std::vector<CoordinateWithSide> possible_farmer_positions(
            const CarcassonneGameState& state) {
        std::vector<CoordinateWithSide> positions;
        const TileAction& lta = *state.last_tile_action;
        const Tile& tile = lta.tile;
        const Coordinate& pos = lta.coordinate;

        for (auto& fc : tile.farms) {
            Farm farm = FarmUtil::find_farm(state,
                FarmerConnectionWithCoordinate(fc, pos));
            if (!FarmUtil::has_meeples(state, farm)) {
                if (!fc.farmer_positions.empty()) {
                    positions.push_back(CoordinateWithSide(pos, fc.farmer_positions[0]));
                }
            }
        }
        return positions;
    }
};

} // namespace carcassonne

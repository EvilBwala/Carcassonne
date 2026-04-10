#pragma once
#include <set>
#include <vector>
#include "coordinate.h"
#include "game_state.h"
#include "meeple_position.h"
#include "structures.h"
#include "tile.h"
#include "types.h"

namespace carcassonne {

class RoadUtil {
public:
    static CoordinateWithSide opposite_edge(const CoordinateWithSide& pos) {
        auto& c = pos.coordinate;
        if (pos.side == Side::TOP)
            return CoordinateWithSide(Coordinate(c.row - 1, c.column), Side::BOTTOM);
        if (pos.side == Side::RIGHT)
            return CoordinateWithSide(Coordinate(c.row, c.column + 1), Side::LEFT);
        if (pos.side == Side::BOTTOM)
            return CoordinateWithSide(Coordinate(c.row + 1, c.column), Side::TOP);
        if (pos.side == Side::LEFT)
            return CoordinateWithSide(Coordinate(c.row, c.column - 1), Side::RIGHT);
        return pos;
    }

    static Road find_road(const CarcassonneGameState& state, CoordinateWithSide road_position) {
        std::set<CoordinateWithSide> roads;
        auto initial = outgoing_roads_for_position(state, road_position);
        roads.insert(initial.begin(), initial.end());

        std::set<CoordinateWithSide> open_connections;
        for (auto& r : roads) open_connections.insert(opposite_edge(r));

        std::set<CoordinateWithSide> explored = roads;
        explored.insert(open_connections.begin(), open_connections.end());

        while (!open_connections.empty()) {
            auto it = open_connections.begin();
            CoordinateWithSide oc = *it;
            open_connections.erase(it);

            auto new_roads = outgoing_roads_for_position(state, oc);
            roads.insert(new_roads.begin(), new_roads.end());

            for (auto& nr : new_roads) {
                CoordinateWithSide oe = opposite_edge(nr);
                explored.insert(nr);
                if (explored.find(oe) == explored.end()) {
                    open_connections.insert(oe);
                    explored.insert(oe);
                }
            }
        }

        bool finished = (explored.size() == roads.size());
        return Road(roads, finished);
    }

    static std::vector<CoordinateWithSide> outgoing_roads_for_position(
            const CarcassonneGameState& state, const CoordinateWithSide& pos) {
        const Tile* tile = state.get_tile(pos.coordinate.row, pos.coordinate.column);
        if (!tile) return {};

        std::vector<CoordinateWithSide> roads;
        for (auto& conn : tile->road) {
            if (conn.a == pos.side || conn.b == pos.side) {
                if (conn.a != Side::CENTER)
                    roads.push_back(CoordinateWithSide(pos.coordinate, conn.a));
                if (conn.b != Side::CENTER)
                    roads.push_back(CoordinateWithSide(pos.coordinate, conn.b));
            }
        }
        return roads;
    }

    static bool road_contains_meeples(const CarcassonneGameState& state, const Road& road) {
        for (auto& rp : road.road_positions) {
            for (int i = 0; i < state.players; ++i) {
                for (auto& mp : state.placed_meeples[i]) {
                    if (rp == mp.coordinate_with_side) return true;
                }
            }
        }
        return false;
    }

    static std::vector<std::vector<MeeplePosition>> find_meeples(
            const CarcassonneGameState& state, const Road& road) {
        std::vector<std::vector<MeeplePosition>> meeples(state.players);
        for (auto& rp : road.road_positions) {
            for (int i = 0; i < state.players; ++i) {
                for (auto& mp : state.placed_meeples[i]) {
                    if (rp == mp.coordinate_with_side)
                        meeples[i].push_back(mp);
                }
            }
        }
        return meeples;
    }

    static std::vector<Road> find_roads(const CarcassonneGameState& state, const Coordinate& coord) {
        std::set<Road> roads;
        const Tile* tile = state.get_tile(coord.row, coord.column);
        if (!tile) return {};

        for (int side : {Side::TOP, Side::RIGHT, Side::BOTTOM, Side::LEFT}) {
            if (tile->get_type(side) == TerrainType::ROAD) {
                Road r = find_road(state, CoordinateWithSide(coord, side));
                roads.insert(r);
            }
        }
        return std::vector<Road>(roads.begin(), roads.end());
    }
};

} // namespace carcassonne

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

class CityUtil {
public:
    static City find_city(const CarcassonneGameState& state, CoordinateWithSide city_position) {
        std::set<CoordinateWithSide> cities;
        auto initial = cities_for_position(state, city_position);
        cities.insert(initial.begin(), initial.end());

        std::set<CoordinateWithSide> open_edges;
        for (auto& c : cities) open_edges.insert(opposite_edge(c));

        std::set<CoordinateWithSide> explored = cities;
        explored.insert(open_edges.begin(), open_edges.end());

        while (!open_edges.empty()) {
            auto it = open_edges.begin();
            CoordinateWithSide open_edge = *it;
            open_edges.erase(it);

            auto new_cities = cities_for_position(state, open_edge);
            cities.insert(new_cities.begin(), new_cities.end());

            for (auto& nc : new_cities) {
                CoordinateWithSide oe = opposite_edge(nc);
                explored.insert(nc);
                if (explored.find(oe) == explored.end()) {
                    open_edges.insert(oe);
                    explored.insert(oe);
                }
            }
        }

        bool finished = (explored.size() == cities.size());
        return City(cities, finished);
    }

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

    static std::vector<CoordinateWithSide> cities_for_position(
            const CarcassonneGameState& state, const CoordinateWithSide& pos) {
        std::vector<CoordinateWithSide> result;
        const Tile* tile = state.get_tile(pos.coordinate.row, pos.coordinate.column);
        if (!tile) return result;

        for (auto& city_group : tile->city) {
            bool found = false;
            for (int s : city_group) {
                if (s == pos.side) { found = true; break; }
            }
            if (found) {
                for (int s : city_group) {
                    result.push_back(CoordinateWithSide(pos.coordinate, s));
                }
            }
        }
        return result;
    }

    static bool city_contains_meeples(const CarcassonneGameState& state, const City& city) {
        for (auto& cp : city.city_positions) {
            for (int i = 0; i < state.players; ++i) {
                for (auto& mp : state.placed_meeples[i]) {
                    if (cp == mp.coordinate_with_side) return true;
                }
            }
        }
        return false;
    }

    static std::vector<std::vector<MeeplePosition>> find_meeples(
            const CarcassonneGameState& state, const City& city) {
        std::vector<std::vector<MeeplePosition>> meeples(state.players);
        for (auto& cp : city.city_positions) {
            for (int i = 0; i < state.players; ++i) {
                for (auto& mp : state.placed_meeples[i]) {
                    if (cp == mp.coordinate_with_side)
                        meeples[i].push_back(mp);
                }
            }
        }
        return meeples;
    }

    static std::vector<City> find_cities(const CarcassonneGameState& state,
            const Coordinate& coord,
            std::vector<int> sides = {Side::TOP, Side::RIGHT, Side::BOTTOM, Side::LEFT}) {
        std::set<City> cities;
        const Tile* tile = state.get_tile(coord.row, coord.column);
        if (!tile) return {};

        for (int side : sides) {
            if (tile->get_type(side) == TerrainType::CITY) {
                City c = find_city(state, CoordinateWithSide(coord, side));
                cities.insert(c);
            }
        }
        return std::vector<City>(cities.begin(), cities.end());
    }
};

} // namespace carcassonne

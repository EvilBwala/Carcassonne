#pragma once
#include <algorithm>
#include <numeric>
#include <set>
#include <vector>
#include "city_util.h"
#include "coordinate.h"
#include "farm_util.h"
#include "game_state.h"
#include "meeple_position.h"
#include "meeple_util.h"
#include "road_util.h"
#include "structures.h"
#include "tile.h"
#include "types.h"

namespace carcassonne {

class PointsCollector {
public:
    static void remove_meeples_and_collect_points(CarcassonneGameState& state,
                                                    const Coordinate& coordinate) {
        // Cities
        auto cities = CityUtil::find_cities(state, coordinate);
        for (auto& city : cities) {
            if (!city.finished) continue;
            auto meeples = CityUtil::find_meeples(state, city);
            auto counts = get_meeple_counts_per_player(meeples);
            int total = std::accumulate(counts.begin(), counts.end(), 0);
            if (total == 0) continue;
            auto winner = get_winning_player(counts);
            if (winner.has_value()) {
                int points = count_city_points(state, city);
                state.scores[*winner] += points;
            }
            MeepleUtil::remove_meeples(state, meeples);
        }

        // Roads
        auto roads = RoadUtil::find_roads(state, coordinate);
        for (auto& road : roads) {
            if (!road.finished) continue;
            auto meeples = RoadUtil::find_meeples(state, road);
            auto counts = get_meeple_counts_per_player(meeples);
            int total = std::accumulate(counts.begin(), counts.end(), 0);
            if (total == 0) continue;
            auto winner = get_winning_player(counts);
            if (winner.has_value()) {
                int points = count_road_points(state, road);
                state.scores[*winner] += points;
            }
            MeepleUtil::remove_meeples(state, meeples);
        }

        // Chapels/flowers
        for (int row = coordinate.row - 1; row <= coordinate.row + 1; ++row) {
            for (int col = coordinate.column - 1; col <= coordinate.column + 1; ++col) {
                const Tile* tile = state.get_tile(row, col);
                if (!tile) continue;
                Coordinate coord(row, col);
                CoordinateWithSide cws(coord, Side::CENTER);
                auto meeple_player = MeepleUtil::position_contains_meeple(state, cws);
                if ((tile->chapel || tile->flowers) && meeple_player.has_value()) {
                    int points = chapel_or_flowers_points(state, coord);
                    if (points == 9) {
                        state.scores[*meeple_player] += points;
                        std::vector<std::vector<MeeplePosition>> meeples_pp(state.players);
                        for (auto& mp : state.placed_meeples[*meeple_player]) {
                            if (cws == mp.coordinate_with_side) {
                                meeples_pp[*meeple_player].push_back(mp);
                            }
                        }
                        MeepleUtil::remove_meeples(state, meeples_pp);
                    }
                }
            }
        }
    }

    static std::optional<int> get_winning_player(const std::vector<int>& counts) {
        int max_val = *std::max_element(counts.begin(), counts.end());
        int num_winners = 0;
        int winner = -1;
        for (int i = 0; i < (int)counts.size(); ++i) {
            if (counts[i] == max_val) {
                num_winners++;
                winner = i;
            }
        }
        if (num_winners == 1) return winner;
        return std::nullopt;
    }

    static int count_city_points(const CarcassonneGameState& state, const City& city) {
        int points = 0;
        bool has_cathedral = false;

        std::set<Coordinate> coordinates;
        for (auto& pos : city.city_positions) {
            const Tile* tile = state.get_tile(pos.coordinate.row, pos.coordinate.column);
            if (tile && tile->cathedral) has_cathedral = true;
            coordinates.insert(pos.coordinate);
        }

        if (!city.finished && has_cathedral) return 0;

        for (auto& coord : coordinates) {
            const Tile* tile = state.get_tile(coord.row, coord.column);
            if (!tile) continue;
            if (tile->shield) {
                if (has_cathedral) points += 6;
                else points += city.finished ? 4 : 2;
            } else {
                if (has_cathedral) points += 3;
                else points += city.finished ? 2 : 1;
            }
        }
        return points;
    }

    static int count_road_points(const CarcassonneGameState& state, const Road& road) {
        bool has_inn = false;
        std::set<Coordinate> coordinates;
        for (auto& pos : road.road_positions) {
            const Tile* tile = state.get_tile(pos.coordinate.row, pos.coordinate.column);
            if (tile && !tile->inn.empty()) has_inn = true;
            coordinates.insert(pos.coordinate);
        }

        if (!road.finished && has_inn) return 0;
        return has_inn ? (int)coordinates.size() * 2 : (int)coordinates.size();
    }

    static int chapel_or_flowers_points(const CarcassonneGameState& state,
                                         const Coordinate& coordinate) {
        int points = 0;
        for (int row = coordinate.row - 1; row <= coordinate.row + 1; ++row) {
            for (int col = coordinate.column - 1; col <= coordinate.column + 1; ++col) {
                if (state.get_tile(row, col)) points++;
            }
        }
        return points;
    }

    static void count_final_scores(CarcassonneGameState& state) {
        for (int player = 0; player < state.players; ++player) {
            std::set<MeeplePosition> meeples_to_remove(
                state.placed_meeples[player].begin(),
                state.placed_meeples[player].end());

            while (!meeples_to_remove.empty()) {
                auto it = meeples_to_remove.begin();
                MeeplePosition mp = *it;
                meeples_to_remove.erase(it);

                const Tile* tile = state.get_tile(
                    mp.coordinate_with_side.coordinate.row,
                    mp.coordinate_with_side.coordinate.column);
                if (!tile) continue;

                TerrainType tt = tile->get_type(mp.coordinate_with_side.side);

                if (tt == TerrainType::CITY) {
                    City city = CityUtil::find_city(state, mp.coordinate_with_side);
                    auto meeples = CityUtil::find_meeples(state, city);
                    auto counts = get_meeple_counts_per_player(meeples);
                    auto winner = get_winning_player(counts);
                    if (winner.has_value()) {
                        state.scores[*winner] += count_city_points(state, city);
                    }
                    MeepleUtil::remove_meeples(state, meeples);
                    continue;
                }

                if (tt == TerrainType::ROAD) {
                    Road road = RoadUtil::find_road(state, mp.coordinate_with_side);
                    auto meeples = RoadUtil::find_meeples(state, road);
                    auto counts = get_meeple_counts_per_player(meeples);
                    auto winner = get_winning_player(counts);
                    if (winner.has_value()) {
                        state.scores[*winner] += count_road_points(state, road);
                    }
                    MeepleUtil::remove_meeples(state, meeples);
                    continue;
                }

                if (tt == TerrainType::CHAPEL || tt == TerrainType::FLOWERS) {
                    int pts = chapel_or_flowers_points(state, mp.coordinate_with_side.coordinate);
                    state.scores[player] += pts;
                    std::vector<std::vector<MeeplePosition>> mpp(state.players);
                    mpp[player].push_back(mp);
                    MeepleUtil::remove_meeples(state, mpp);
                    continue;
                }

                if (mp.meeple_type == MeepleType::FARMER || mp.meeple_type == MeepleType::BIG_FARMER) {
                    Farm farm = FarmUtil::find_farm_by_coordinate(state, mp.coordinate_with_side);
                    auto meeples = FarmUtil::find_meeples(state, farm);
                    auto counts = get_meeple_counts_per_player(meeples);
                    auto winner = get_winning_player(counts);
                    if (winner.has_value()) {
                        state.scores[*winner] += count_farm_points(state, farm);
                    }
                    MeepleUtil::remove_meeples(state, meeples);
                    continue;
                }
            }
        }
    }

    static std::vector<int> get_meeple_counts_per_player(
            const std::vector<std::vector<MeeplePosition>>& meeples) {
        std::vector<int> counts;
        counts.reserve(meeples.size());
        for (auto& player_meeples : meeples) {
            int count = 0;
            for (auto& mp : player_meeples) {
                count += (mp.meeple_type == MeepleType::BIG || mp.meeple_type == MeepleType::BIG_FARMER) ? 2 : 1;
            }
            counts.push_back(count);
        }
        return counts;
    }

    static int count_farm_points(const CarcassonneGameState& state, const Farm& farm) {
        std::set<City> cities;

        for (auto& fcwc : farm.farmer_connections_with_coordinate) {
            auto found = CityUtil::find_cities(state, fcwc.coordinate,
                                                fcwc.farmer_connection.city_sides);
            for (auto& c : found) cities.insert(c);
        }

        int points = 0;
        for (auto& city : cities) {
            if (city.finished) points += 3;
        }
        return points;
    }
};

} // namespace carcassonne

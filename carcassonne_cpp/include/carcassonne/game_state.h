#pragma once
#include <algorithm>
#include <optional>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "actions.h"
#include "coordinate.h"
#include "meeple_position.h"
#include "tile.h"
#include "tile_sets.h"
#include "types.h"

namespace carcassonne {

class TileRegistry {
public:
    std::unordered_map<std::string, Tile> tile_dict;

    void add_deck(const std::unordered_map<std::string, Tile>& deck) {
        for (auto& [desc, tile] : deck) {
            tile_dict[desc] = tile;
        }
    }

    std::unordered_map<std::string, std::string> tile_to_json(const Tile& tile) const {
        return {{"description", tile.description}, {"turns", std::to_string(tile.turns)}};
    }

    Tile tile_from_json(const std::string& description, int turns) const {
        auto it = tile_dict.find(description);
        if (it == tile_dict.end()) {
            throw std::runtime_error("Unknown tile: " + description);
        }
        return it->second.turn(turns);
    }
};

TileRegistry& get_global_tile_registry();

struct PairHash {
    size_t operator()(const std::pair<int,int>& p) const {
        return std::hash<int>{}(p.first) ^ (std::hash<int>{}(p.second) << 16);
    }
};

class CarcassonneGameState {
public:
    std::vector<Tile> deck;
    std::vector<SupplementaryRule> supplementary_rules;
    std::unordered_map<std::pair<int,int>, Tile, PairHash> board;
    Coordinate starting_position;
    std::optional<Tile> next_tile;
    int players = 2;
    std::vector<int> meeples;
    std::vector<int> abbots;
    std::vector<int> big_meeples;
    std::vector<std::vector<MeeplePosition>> placed_meeples;
    std::vector<int> scores;
    int current_player = 0;
    GamePhase phase = GamePhase::TILES;
    std::optional<TileAction> last_tile_action;
    Rotation last_river_rotation = Rotation::NONE;

    CarcassonneGameState() = default;

    CarcassonneGameState(
        std::vector<TileSet> tile_sets,
        std::vector<SupplementaryRule> supp_rules,
        int num_players = 2,
        Coordinate start_pos = Coordinate(0, 0)
    ) {
        supplementary_rules = std::move(supp_rules);
        starting_position = start_pos;
        players = num_players;

        deck = initialize_deck(tile_sets);
        if (!deck.empty()) {
            next_tile = deck.front();
            deck.erase(deck.begin());
        }

        meeples.assign(players, 7);
        scores.assign(players, 0);
        current_player = 0;
        phase = GamePhase::TILES;

        bool has_abbots = std::find(supplementary_rules.begin(), supplementary_rules.end(),
                                     SupplementaryRule::ABBOTS) != supplementary_rules.end();
        abbots.assign(players, has_abbots ? 1 : 0);

        bool has_ic = std::find(tile_sets.begin(), tile_sets.end(),
                                TileSet::INNS_AND_CATHEDRALS) != tile_sets.end();
        big_meeples.assign(players, has_ic ? 1 : 0);

        placed_meeples.resize(players);
    }

    const Tile* get_tile(int row, int col) const {
        auto it = board.find({row, col});
        if (it != board.end()) return &it->second;
        return nullptr;
    }

    bool empty_board() const { return board.empty(); }
    bool is_terminated() const { return !next_tile.has_value(); }

    CarcassonneGameState simple_copy() const {
        CarcassonneGameState ns;
        ns.deck = deck;
        ns.supplementary_rules = supplementary_rules;
        ns.board = board; // shared until modified
        ns.starting_position = starting_position;
        ns.next_tile = next_tile;
        ns.players = players;
        ns.meeples = meeples;
        ns.abbots = abbots;
        ns.big_meeples = big_meeples;
        ns.placed_meeples = placed_meeples;
        ns.scores = scores;
        ns.current_player = current_player;
        ns.phase = phase;
        ns.last_tile_action = last_tile_action;
        ns.last_river_rotation = last_river_rotation;
        return ns;
    }

    std::vector<Tile> initialize_deck(const std::vector<TileSet>& tile_sets) {
        std::vector<Tile> result;

        bool has_river = std::find(tile_sets.begin(), tile_sets.end(), TileSet::THE_RIVER) != tile_sets.end();
        if (has_river) {
            auto& rd = get_the_river_deck();
            result.push_back(rd.tiles.at("river_start"));

            std::vector<Tile> river_middle;
            for (auto& [name, count] : rd.tile_counts) {
                if (name == "river_start" || name == "river_end") continue;
                auto it = rd.tiles.find(name);
                if (it != rd.tiles.end()) {
                    for (int i = 0; i < count; ++i)
                        river_middle.push_back(it->second);
                }
            }
            std::random_device rdev;
            std::mt19937 rng(rdev());
            std::shuffle(river_middle.begin(), river_middle.end(), rng);
            for (auto& t : river_middle) result.push_back(std::move(t));
            result.push_back(rd.tiles.at("river_end"));
        }

        std::vector<Tile> main_tiles;

        bool has_base = std::find(tile_sets.begin(), tile_sets.end(), TileSet::BASE) != tile_sets.end();
        if (has_base) {
            auto& bd = get_base_deck();
            for (auto& [name, count] : bd.tile_counts) {
                auto it = bd.tiles.find(name);
                if (it != bd.tiles.end()) {
                    for (int i = 0; i < count; ++i)
                        main_tiles.push_back(it->second);
                }
            }
        }

        bool has_ic = std::find(tile_sets.begin(), tile_sets.end(), TileSet::INNS_AND_CATHEDRALS) != tile_sets.end();
        if (has_ic) {
            auto& icd = get_inns_and_cathedrals_deck();
            for (auto& [name, count] : icd.tile_counts) {
                auto it = icd.tiles.find(name);
                if (it != icd.tiles.end()) {
                    for (int i = 0; i < count; ++i)
                        main_tiles.push_back(it->second);
                }
            }
        }

        std::random_device rdev;
        std::mt19937 rng(rdev());
        std::shuffle(main_tiles.begin(), main_tiles.end(), rng);
        for (auto& t : main_tiles) result.push_back(std::move(t));

        return result;
    }

};

} // namespace carcassonne

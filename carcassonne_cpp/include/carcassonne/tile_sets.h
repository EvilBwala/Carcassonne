#pragma once
#include <string>
#include <unordered_map>
#include <vector>
#include <utility>
#include "tile.h"

namespace carcassonne {

struct TileDeck {
    std::unordered_map<std::string, Tile> tiles;
    std::unordered_map<std::string, int> tile_counts;
};

const TileDeck& get_base_deck();
const TileDeck& get_inns_and_cathedrals_deck();
const TileDeck& get_the_river_deck();

} // namespace carcassonne

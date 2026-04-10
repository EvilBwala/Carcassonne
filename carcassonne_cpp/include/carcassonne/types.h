#pragma once
#include <string>
#include <vector>
#include <functional>

namespace carcassonne {

namespace Side {
    constexpr int TOP = 0;
    constexpr int TOP_RIGHT = 1;
    constexpr int RIGHT = 2;
    constexpr int BOTTOM_RIGHT = 3;
    constexpr int BOTTOM = 4;
    constexpr int BOTTOM_LEFT = 5;
    constexpr int LEFT = 6;
    constexpr int TOP_LEFT = 7;
    constexpr int CENTER = 8;
}

namespace MeepleType {
    constexpr int NORMAL = 0;
    constexpr int ABBOT = 1;
    constexpr int FARMER = 2;
    constexpr int BIG = 3;
    constexpr int BIG_FARMER = 4;
}

enum class GamePhase {
    TILES,
    MEEPLES
};

inline std::string game_phase_to_string(GamePhase p) {
    return p == GamePhase::TILES ? "tiles" : "meeples";
}

inline GamePhase game_phase_from_string(const std::string& s) {
    return s == "tiles" ? GamePhase::TILES : GamePhase::MEEPLES;
}

enum class TerrainType {
    CITY,
    GRASS,
    ROAD,
    CHAPEL,
    FLOWERS,
    UNPLAYABLE
};

inline std::string terrain_type_to_string(TerrainType t) {
    switch (t) {
        case TerrainType::CITY: return "city";
        case TerrainType::GRASS: return "grass";
        case TerrainType::ROAD: return "road";
        case TerrainType::CHAPEL: return "chapel";
        case TerrainType::FLOWERS: return "flowers";
        case TerrainType::UNPLAYABLE: return "unplayable";
    }
    return "unknown";
}

enum class Rotation {
    CLOCKWISE,
    COUNTER_CLOCKWISE,
    NONE
};

inline std::string rotation_to_string(Rotation r) {
    switch (r) {
        case Rotation::CLOCKWISE: return "clockwise";
        case Rotation::COUNTER_CLOCKWISE: return "counter_clockwise";
        case Rotation::NONE: return "none";
    }
    return "none";
}

inline Rotation rotation_from_string(const std::string& s) {
    if (s == "clockwise") return Rotation::CLOCKWISE;
    if (s == "counter_clockwise") return Rotation::COUNTER_CLOCKWISE;
    return Rotation::NONE;
}

namespace FarmerSide {
    constexpr int TLL = 0;
    constexpr int TLT = 1;
    constexpr int TRT = 2;
    constexpr int TRR = 3;
    constexpr int BRR = 4;
    constexpr int BRB = 5;
    constexpr int BLB = 6;
    constexpr int BLL = 7;

    inline int get_side(int farmer_side) {
        switch (farmer_side) {
            case TLL: case BLL: return Side::LEFT;
            case TLT: case TRT: return Side::TOP;
            case TRR: case BRR: return Side::RIGHT;
            case BRB: case BLB: return Side::BOTTOM;
        }
        return -1;
    }
}

enum class SupplementaryRule {
    FARMERS,
    ABBOTS,
    NORMAL_MEEPLES_CAN_USE_FLOWERS
};

inline std::string supplementary_rule_to_string(SupplementaryRule r) {
    switch (r) {
        case SupplementaryRule::FARMERS: return "farmers";
        case SupplementaryRule::ABBOTS: return "abbots";
        case SupplementaryRule::NORMAL_MEEPLES_CAN_USE_FLOWERS: return "normal_meeples_can_use_flowers";
    }
    return "unknown";
}

inline SupplementaryRule supplementary_rule_from_string(const std::string& s) {
    if (s == "farmers") return SupplementaryRule::FARMERS;
    if (s == "abbots") return SupplementaryRule::ABBOTS;
    return SupplementaryRule::NORMAL_MEEPLES_CAN_USE_FLOWERS;
}

enum class TileSet {
    BASE,
    THE_RIVER,
    INNS_AND_CATHEDRALS
};

inline std::string tile_set_to_string(TileSet t) {
    switch (t) {
        case TileSet::BASE: return "base";
        case TileSet::THE_RIVER: return "the_river";
        case TileSet::INNS_AND_CATHEDRALS: return "inns_and_cathedrals";
    }
    return "unknown";
}

} // namespace carcassonne

#pragma once
#include "game_state.h"
#include "tile.h"
#include "types.h"

namespace carcassonne {

class TileFitter {
public:
    static bool grass_fits(const Tile& center, const Tile* top, const Tile* right,
                           const Tile* bottom, const Tile* left) {
        for (int side : center.grass) {
            if (side == Side::LEFT && left) {
                bool found = false;
                for (int s : left->grass) { if (s == Side::RIGHT) { found = true; break; } }
                if (!found) return false;
            }
            if (side == Side::RIGHT && right) {
                bool found = false;
                for (int s : right->grass) { if (s == Side::LEFT) { found = true; break; } }
                if (!found) return false;
            }
            if (side == Side::TOP && top) {
                bool found = false;
                for (int s : top->grass) { if (s == Side::BOTTOM) { found = true; break; } }
                if (!found) return false;
            }
            if (side == Side::BOTTOM && bottom) {
                bool found = false;
                for (int s : bottom->grass) { if (s == Side::TOP) { found = true; break; } }
                if (!found) return false;
            }
        }
        return true;
    }

    static bool cities_fit(const Tile& center, const Tile* top, const Tile* right,
                           const Tile* bottom, const Tile* left) {
        auto city_sides = center.get_city_sides();
        for (int side : city_sides) {
            if (side == Side::LEFT && left && !left->get_city_sides().count(Side::RIGHT)) return false;
            if (side == Side::RIGHT && right && !right->get_city_sides().count(Side::LEFT)) return false;
            if (side == Side::TOP && top && !top->get_city_sides().count(Side::BOTTOM)) return false;
            if (side == Side::BOTTOM && bottom && !bottom->get_city_sides().count(Side::TOP)) return false;
        }
        return true;
    }

    static bool roads_fit(const Tile& center, const Tile* top, const Tile* right,
                          const Tile* bottom, const Tile* left) {
        auto road_ends = center.get_road_ends();
        for (int side : road_ends) {
            if (side == Side::LEFT && left && !left->get_road_ends().count(Side::RIGHT)) return false;
            if (side == Side::RIGHT && right && !right->get_road_ends().count(Side::LEFT)) return false;
            if (side == Side::TOP && top && !top->get_road_ends().count(Side::BOTTOM)) return false;
            if (side == Side::BOTTOM && bottom && !bottom->get_road_ends().count(Side::TOP)) return false;
        }
        return true;
    }

    static bool rivers_fit(const Tile& center, const Tile* top, const Tile* right,
                           const Tile* bottom, const Tile* left,
                           const CarcassonneGameState* state = nullptr) {
        auto river_ends = center.get_river_ends();
        if (river_ends.empty()) return true;

        for (int side : river_ends) {
            if (side == Side::LEFT && left && !left->get_river_ends().count(Side::RIGHT)) return false;
            if (side == Side::RIGHT && right && !right->get_river_ends().count(Side::LEFT)) return false;
            if (side == Side::TOP && top && !top->get_river_ends().count(Side::BOTTOM)) return false;
            if (side == Side::BOTTOM && bottom && !bottom->get_river_ends().count(Side::TOP)) return false;
        }

        bool has_connected = false;
        for (int side : river_ends) {
            if (side == Side::LEFT && left && left->get_river_ends().count(Side::RIGHT)) has_connected = true;
            if (side == Side::RIGHT && right && right->get_river_ends().count(Side::LEFT)) has_connected = true;
            if (side == Side::TOP && top && top->get_river_ends().count(Side::BOTTOM)) has_connected = true;
            if (side == Side::BOTTOM && bottom && bottom->get_river_ends().count(Side::TOP)) has_connected = true;
        }
        if (!has_connected) return false;

        return true;
    }

    static bool fits(const Tile& center, const Tile* top, const Tile* right,
                     const Tile* bottom, const Tile* left,
                     const CarcassonneGameState* state = nullptr) {
        if (!top && !right && !bottom && !left) return false;

        return grass_fits(center, top, right, bottom, left)
            && cities_fit(center, top, right, bottom, left)
            && roads_fit(center, top, right, bottom, left)
            && rivers_fit(center, top, right, bottom, left, state);
    }
};

} // namespace carcassonne

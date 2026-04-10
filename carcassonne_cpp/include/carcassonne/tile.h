#pragma once
#include <string>
#include <vector>
#include <set>
#include <algorithm>
#include "types.h"
#include "connection.h"
#include "side_modification_util.h"

namespace carcassonne {

class Tile {
public:
    std::string description;
    int turns = 0;
    std::vector<Connection> road;
    std::vector<Connection> river;
    std::vector<std::vector<int>> city;
    std::vector<int> grass;
    std::vector<FarmerConnection> farms;
    bool shield = false;
    bool chapel = false;
    bool flowers = false;
    std::vector<int> inn;
    bool cathedral = false;
    std::vector<int> unplayable_sides;
    std::string image;

    Tile() = default;

    std::set<int> get_road_ends() const {
        std::set<int> sides;
        for (auto& r : road) {
            sides.insert(r.a);
            sides.insert(r.b);
        }
        return sides;
    }

    std::set<int> get_river_ends() const {
        std::set<int> sides;
        for (auto& r : river) {
            sides.insert(r.a);
            sides.insert(r.b);
        }
        return sides;
    }

    std::set<int> get_city_sides() const {
        std::set<int> sides;
        for (auto& group : city) {
            for (int s : group) sides.insert(s);
        }
        return sides;
    }

    bool has_river() const { return !river.empty(); }

    TerrainType get_type(int side) const {
        if (std::find(unplayable_sides.begin(), unplayable_sides.end(), side) != unplayable_sides.end())
            return TerrainType::UNPLAYABLE;
        if (side == Side::CENTER && chapel) return TerrainType::CHAPEL;
        if (side == Side::CENTER && flowers) return TerrainType::FLOWERS;
        if (get_river_ends().count(side)) return TerrainType::UNPLAYABLE;
        if (get_road_ends().count(side)) return TerrainType::ROAD;
        if (get_city_sides().count(side)) return TerrainType::CITY;
        if (std::find(grass.begin(), grass.end(), side) != grass.end()) return TerrainType::GRASS;
        return TerrainType::GRASS; // default fallback
    }

    Tile turn(int times) const {
        Tile t;
        t.description = description;
        t.turns = times;
        t.shield = shield;
        t.chapel = chapel;
        t.flowers = flowers;
        t.cathedral = cathedral;
        t.image = image;

        for (auto& r : road)
            t.road.push_back(SideModificationUtil::turn_connection(r, times));
        for (auto& r : river)
            t.river.push_back(SideModificationUtil::turn_connection(r, times));
        for (auto& group : city)
            t.city.push_back(SideModificationUtil::turn_sides(group, times));
        for (int s : grass)
            t.grass.push_back(SideModificationUtil::turn_side(s, times));
        for (auto& fc : farms)
            t.farms.push_back(SideModificationUtil::turn_farmer_connection(fc, times));
        for (int s : inn)
            t.inn.push_back(SideModificationUtil::turn_side(s, times));
        for (int s : unplayable_sides)
            t.unplayable_sides.push_back(SideModificationUtil::turn_side(s, times));

        return t;
    }

    bool operator==(const Tile& o) const {
        return description == o.description && turns == o.turns;
    }
    bool operator!=(const Tile& o) const { return !(*this == o); }
};

} // namespace carcassonne

namespace std {
    template<> struct hash<carcassonne::Tile> {
        size_t operator()(const carcassonne::Tile& t) const {
            size_t h = std::hash<std::string>{}(t.description);
            h ^= std::hash<int>{}(t.turns) + 0x9e3779b9 + (h << 6) + (h >> 2);
            return h;
        }
    };
}

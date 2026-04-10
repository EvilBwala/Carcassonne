#pragma once
#include <vector>
#include <set>
#include <functional>
#include "coordinate.h"
#include "connection.h"

namespace carcassonne {

struct City {
    std::set<CoordinateWithSide> city_positions;
    bool finished;

    City() : finished(false) {}
    City(std::set<CoordinateWithSide> cp, bool f) : city_positions(std::move(cp)), finished(f) {}

    bool operator==(const City& o) const {
        return city_positions == o.city_positions && finished == o.finished;
    }
    bool operator!=(const City& o) const { return !(*this == o); }
    bool operator<(const City& o) const {
        if (city_positions != o.city_positions) return city_positions < o.city_positions;
        return finished < o.finished;
    }
};

struct Road {
    std::set<CoordinateWithSide> road_positions;
    bool finished;

    Road() : finished(false) {}
    Road(std::set<CoordinateWithSide> rp, bool f) : road_positions(std::move(rp)), finished(f) {}

    bool operator==(const Road& o) const {
        return road_positions == o.road_positions && finished == o.finished;
    }
    bool operator!=(const Road& o) const { return !(*this == o); }
    bool operator<(const Road& o) const {
        if (road_positions != o.road_positions) return road_positions < o.road_positions;
        return finished < o.finished;
    }
};

struct Farm {
    std::set<FarmerConnectionWithCoordinate> farmer_connections_with_coordinate;

    Farm() = default;
    Farm(std::set<FarmerConnectionWithCoordinate> fcwc)
        : farmer_connections_with_coordinate(std::move(fcwc)) {}
};

struct PlayingPosition {
    Coordinate coordinate;
    int turns;

    PlayingPosition() : turns(0) {}
    PlayingPosition(Coordinate c, int t) : coordinate(c), turns(t) {}
};

} // namespace carcassonne

namespace std {
    template<> struct hash<carcassonne::City> {
        size_t operator()(const carcassonne::City& c) const {
            size_t h = std::hash<bool>{}(c.finished);
            for (auto& p : c.city_positions) {
                h ^= std::hash<carcassonne::CoordinateWithSide>{}(p) + 0x9e3779b9 + (h << 6) + (h >> 2);
            }
            return h;
        }
    };
    template<> struct hash<carcassonne::Road> {
        size_t operator()(const carcassonne::Road& r) const {
            size_t h = std::hash<bool>{}(r.finished);
            for (auto& p : r.road_positions) {
                h ^= std::hash<carcassonne::CoordinateWithSide>{}(p) + 0x9e3779b9 + (h << 6) + (h >> 2);
            }
            return h;
        }
    };
}

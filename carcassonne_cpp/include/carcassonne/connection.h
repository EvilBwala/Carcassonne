#pragma once
#include <vector>
#include <functional>
#include "coordinate.h"

namespace carcassonne {

struct Connection {
    int a;
    int b;

    Connection() : a(0), b(0) {}
    Connection(int a_, int b_) : a(a_), b(b_) {}

    bool operator==(const Connection& o) const { return a == o.a && b == o.b; }
    bool operator!=(const Connection& o) const { return !(*this == o); }
    bool operator<(const Connection& o) const {
        if (a != o.a) return a < o.a;
        return b < o.b;
    }
};

struct FarmerConnection {
    std::vector<int> farmer_positions;
    std::vector<int> tile_connections;
    std::vector<int> city_sides;

    FarmerConnection() = default;
    FarmerConnection(std::vector<int> fp, std::vector<int> tc = {}, std::vector<int> cs = {})
        : farmer_positions(std::move(fp)), tile_connections(std::move(tc)), city_sides(std::move(cs)) {}

    bool operator==(const FarmerConnection& o) const {
        return farmer_positions == o.farmer_positions
            && tile_connections == o.tile_connections
            && city_sides == o.city_sides;
    }
    bool operator!=(const FarmerConnection& o) const { return !(*this == o); }
    bool operator<(const FarmerConnection& o) const {
        if (farmer_positions != o.farmer_positions) return farmer_positions < o.farmer_positions;
        if (tile_connections != o.tile_connections) return tile_connections < o.tile_connections;
        return city_sides < o.city_sides;
    }
};

struct FarmerConnectionWithCoordinate {
    FarmerConnection farmer_connection;
    Coordinate coordinate;

    FarmerConnectionWithCoordinate() = default;
    FarmerConnectionWithCoordinate(FarmerConnection fc, Coordinate c)
        : farmer_connection(std::move(fc)), coordinate(c) {}

    bool operator==(const FarmerConnectionWithCoordinate& o) const {
        return farmer_connection == o.farmer_connection && coordinate == o.coordinate;
    }
    bool operator!=(const FarmerConnectionWithCoordinate& o) const { return !(*this == o); }
    bool operator<(const FarmerConnectionWithCoordinate& o) const {
        if (coordinate != o.coordinate) return coordinate < o.coordinate;
        return farmer_connection < o.farmer_connection;
    }
};

} // namespace carcassonne

namespace std {
    template<> struct hash<carcassonne::Connection> {
        size_t operator()(const carcassonne::Connection& c) const {
            return std::hash<int>{}(c.a) ^ (std::hash<int>{}(c.b) << 16);
        }
    };
    template<> struct hash<carcassonne::FarmerConnection> {
        size_t operator()(const carcassonne::FarmerConnection& fc) const {
            size_t h = 0;
            for (int v : fc.farmer_positions) h ^= std::hash<int>{}(v) + 0x9e3779b9 + (h << 6) + (h >> 2);
            for (int v : fc.tile_connections) h ^= std::hash<int>{}(v) + 0x9e3779b9 + (h << 6) + (h >> 2);
            for (int v : fc.city_sides) h ^= std::hash<int>{}(v) + 0x9e3779b9 + (h << 6) + (h >> 2);
            return h;
        }
    };
    template<> struct hash<carcassonne::FarmerConnectionWithCoordinate> {
        size_t operator()(const carcassonne::FarmerConnectionWithCoordinate& fc) const {
            size_t h1 = std::hash<carcassonne::FarmerConnection>{}(fc.farmer_connection);
            size_t h2 = std::hash<carcassonne::Coordinate>{}(fc.coordinate);
            return h1 ^ (h2 << 16);
        }
    };
}

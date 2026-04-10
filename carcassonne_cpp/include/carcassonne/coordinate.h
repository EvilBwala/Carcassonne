#pragma once
#include <functional>
#include <string>
#include <vector>

namespace carcassonne {

struct Coordinate {
    int row;
    int column;

    Coordinate() : row(0), column(0) {}
    Coordinate(int r, int c) : row(r), column(c) {}

    bool operator==(const Coordinate& o) const { return row == o.row && column == o.column; }
    bool operator!=(const Coordinate& o) const { return !(*this == o); }
    bool operator<(const Coordinate& o) const {
        if (row != o.row) return row < o.row;
        return column < o.column;
    }

    std::string to_string() const {
        return "(" + std::to_string(row) + ", " + std::to_string(column) + ")";
    }

    std::vector<int> to_json() const { return {row, column}; }

    static Coordinate from_json(const std::vector<int>& data) {
        return Coordinate(data[0], data[1]);
    }
};

struct CoordinateWithSide {
    Coordinate coordinate;
    int side;

    CoordinateWithSide() : side(0) {}
    CoordinateWithSide(Coordinate c, int s) : coordinate(c), side(s) {}

    bool operator==(const CoordinateWithSide& o) const {
        return coordinate == o.coordinate && side == o.side;
    }
    bool operator!=(const CoordinateWithSide& o) const { return !(*this == o); }
    bool operator<(const CoordinateWithSide& o) const {
        if (coordinate != o.coordinate) return coordinate < o.coordinate;
        return side < o.side;
    }

    std::string to_string() const {
        return coordinate.to_string() + " " + std::to_string(side);
    }
};

struct CoordinateWithFarmerSide {
    Coordinate coordinate;
    int farmer_side;

    CoordinateWithFarmerSide() : farmer_side(0) {}
    CoordinateWithFarmerSide(Coordinate c, int fs) : coordinate(c), farmer_side(fs) {}

    bool operator==(const CoordinateWithFarmerSide& o) const {
        return coordinate == o.coordinate && farmer_side == o.farmer_side;
    }
    bool operator!=(const CoordinateWithFarmerSide& o) const { return !(*this == o); }
    bool operator<(const CoordinateWithFarmerSide& o) const {
        if (coordinate != o.coordinate) return coordinate < o.coordinate;
        return farmer_side < o.farmer_side;
    }

    std::string to_string() const {
        return coordinate.to_string() + " " + std::to_string(farmer_side);
    }
};

} // namespace carcassonne

namespace std {
    template<> struct hash<carcassonne::Coordinate> {
        size_t operator()(const carcassonne::Coordinate& c) const {
            size_t h1 = std::hash<int>{}(c.row);
            size_t h2 = std::hash<int>{}(c.column);
            return h1 ^ (h2 << 16);
        }
    };
    template<> struct hash<carcassonne::CoordinateWithSide> {
        size_t operator()(const carcassonne::CoordinateWithSide& c) const {
            size_t h1 = std::hash<carcassonne::Coordinate>{}(c.coordinate);
            size_t h2 = std::hash<int>{}(c.side);
            return h1 ^ (h2 << 16);
        }
    };
    template<> struct hash<carcassonne::CoordinateWithFarmerSide> {
        size_t operator()(const carcassonne::CoordinateWithFarmerSide& c) const {
            size_t h1 = std::hash<carcassonne::Coordinate>{}(c.coordinate);
            size_t h2 = std::hash<int>{}(c.farmer_side);
            return h1 ^ (h2 << 16);
        }
    };
}

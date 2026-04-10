#pragma once
#include "coordinate.h"
#include "types.h"
#include <functional>

namespace carcassonne {

struct MeeplePosition {
    int meeple_type;
    CoordinateWithSide coordinate_with_side;

    MeeplePosition() : meeple_type(0) {}
    MeeplePosition(int mt, CoordinateWithSide cws) : meeple_type(mt), coordinate_with_side(cws) {}

    bool operator==(const MeeplePosition& o) const {
        return meeple_type == o.meeple_type && coordinate_with_side == o.coordinate_with_side;
    }
    bool operator!=(const MeeplePosition& o) const { return !(*this == o); }
    bool operator<(const MeeplePosition& o) const {
        if (meeple_type != o.meeple_type) return meeple_type < o.meeple_type;
        return coordinate_with_side < o.coordinate_with_side;
    }
};

} // namespace carcassonne

namespace std {
    template<> struct hash<carcassonne::MeeplePosition> {
        size_t operator()(const carcassonne::MeeplePosition& mp) const {
            size_t h1 = std::hash<int>{}(mp.meeple_type);
            size_t h2 = std::hash<carcassonne::CoordinateWithSide>{}(mp.coordinate_with_side);
            return h1 ^ (h2 << 16);
        }
    };
}

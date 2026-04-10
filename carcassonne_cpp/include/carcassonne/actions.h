#pragma once
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include "coordinate.h"
#include "tile.h"
#include "types.h"

namespace carcassonne {

constexpr int ACTION_BOARD_SIZE = 10;
constexpr int NUM_MEEPLE_TYPES = 5;
constexpr int NUM_MEEPLE_SIDES = 9;

constexpr int PLACE_TILE_OFFSET = 0;
constexpr int TILE_ACTION_DIM = ACTION_BOARD_SIZE * ACTION_BOARD_SIZE * 4;
constexpr int PLACE_MEEPLE_OFFSET = TILE_ACTION_DIM;
constexpr int MEEPLE_ACTION_DIM = NUM_MEEPLE_SIDES * NUM_MEEPLE_TYPES;
constexpr int REMOVE_ABBOT_OFFSET = PLACE_MEEPLE_OFFSET + MEEPLE_ACTION_DIM;
constexpr int PASS_OFFSET = REMOVE_ABBOT_OFFSET + 1;
constexpr int ACTION_TOTAL_DIM = PASS_OFFSET + 1;

enum class ActionType {
    TILE,
    MEEPLE,
    PASS
};

struct Action {
    virtual ~Action() = default;
    virtual ActionType type() const = 0;
    virtual int get_idx() const = 0;
    virtual std::string to_string() const = 0;

    bool operator==(const Action& o) const { return get_idx() == o.get_idx(); }
    bool operator!=(const Action& o) const { return !(*this == o); }
};

struct TileAction : Action {
    Tile tile;
    Coordinate coordinate;
    int tile_rotations;

    TileAction() : tile_rotations(0) {}
    TileAction(Tile t, Coordinate c, int tr) : tile(std::move(t)), coordinate(c), tile_rotations(tr) {}

    ActionType type() const override { return ActionType::TILE; }

    int get_idx() const override {
        return PLACE_TILE_OFFSET + coordinate.row * ACTION_BOARD_SIZE * 4
             + coordinate.column * 4 + tile_rotations;
    }

    std::string to_string() const override {
        return "TileAction(" + tile.description + ", " + coordinate.to_string()
             + ", " + std::to_string(tile_rotations) + ")";
    }
};

struct MeepleAction : Action {
    int meeple_type;
    CoordinateWithSide coordinate_with_side;
    bool remove;

    MeepleAction() : meeple_type(0), remove(false) {}
    MeepleAction(int mt, CoordinateWithSide cws, bool r = false)
        : meeple_type(mt), coordinate_with_side(cws), remove(r) {}

    ActionType type() const override { return ActionType::MEEPLE; }

    int get_idx() const override {
        if (remove) return REMOVE_ABBOT_OFFSET;
        return PLACE_MEEPLE_OFFSET + coordinate_with_side.side * NUM_MEEPLE_TYPES + meeple_type;
    }

    std::string to_string() const override {
        return "MeepleAction(" + std::to_string(meeple_type) + ", "
             + coordinate_with_side.to_string() + ", "
             + (remove ? "true" : "false") + ")";
    }
};

struct PassAction : Action {
    ActionType type() const override { return ActionType::PASS; }
    int get_idx() const override { return PASS_OFFSET; }
    std::string to_string() const override { return "PassAction()"; }
};

using ActionPtr = std::shared_ptr<Action>;

inline ActionPtr make_tile_action(Tile t, Coordinate c, int tr) {
    return std::make_shared<TileAction>(std::move(t), c, tr);
}

inline ActionPtr make_meeple_action(int mt, CoordinateWithSide cws, bool r = false) {
    return std::make_shared<MeepleAction>(mt, cws, r);
}

inline ActionPtr make_pass_action() {
    return std::make_shared<PassAction>();
}

} // namespace carcassonne

namespace std {
    template<> struct hash<carcassonne::ActionPtr> {
        size_t operator()(const carcassonne::ActionPtr& a) const {
            return a ? std::hash<int>{}(a->get_idx()) : 0;
        }
    };
}

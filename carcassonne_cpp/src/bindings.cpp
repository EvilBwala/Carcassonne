#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include <optional>
#include <sstream>

#include "carcassonne/types.h"
#include "carcassonne/coordinate.h"
#include "carcassonne/connection.h"
#include "carcassonne/tile.h"
#include "carcassonne/meeple_position.h"
#include "carcassonne/structures.h"
#include "carcassonne/actions.h"
#include "carcassonne/game_state.h"
#include "carcassonne/tile_sets.h"
#include "carcassonne/side_modification_util.h"
#include "carcassonne/tile_fitter.h"
#include "carcassonne/tile_position_finder.h"
#include "carcassonne/city_util.h"
#include "carcassonne/road_util.h"
#include "carcassonne/farm_util.h"
#include "carcassonne/meeple_util.h"
#include "carcassonne/points_collector.h"
#include "carcassonne/possible_move_finder.h"
#include "carcassonne/state_updater.h"
#include "carcassonne/action_util.h"
#include "carcassonne/graph_features.h"

namespace py = pybind11;
using namespace carcassonne;

// Wrapper that holds an ActionPtr and acts as either TileAction, MeepleAction, or PassAction
// for Python compatibility
struct PyAction {
    ActionPtr ptr;
    PyAction() : ptr(make_pass_action()) {}
    PyAction(ActionPtr p) : ptr(std::move(p)) {}

    int get_idx() const { return ptr->get_idx(); }
    ActionType type() const { return ptr->type(); }
    std::string to_string() const { return ptr->to_string(); }
    bool operator==(const PyAction& o) const { return get_idx() == o.get_idx(); }
    bool operator!=(const PyAction& o) const { return !(*this == o); }
};

namespace std {
    template<> struct hash<PyAction> {
        size_t operator()(const PyAction& a) const {
            return std::hash<int>{}(a.get_idx());
        }
    };
}

PYBIND11_MODULE(_carcassonne_engine, m) {
    m.doc() = "Carcassonne game engine (C++ implementation)";

    // ---- Side constants ----
    auto side_mod = m.def_submodule("Side");
    side_mod.attr("TOP") = Side::TOP;
    side_mod.attr("TOP_RIGHT") = Side::TOP_RIGHT;
    side_mod.attr("RIGHT") = Side::RIGHT;
    side_mod.attr("BOTTOM_RIGHT") = Side::BOTTOM_RIGHT;
    side_mod.attr("BOTTOM") = Side::BOTTOM;
    side_mod.attr("BOTTOM_LEFT") = Side::BOTTOM_LEFT;
    side_mod.attr("LEFT") = Side::LEFT;
    side_mod.attr("TOP_LEFT") = Side::TOP_LEFT;
    side_mod.attr("CENTER") = Side::CENTER;


    // ---- MeepleType constants ----
    auto mt_mod = m.def_submodule("MeepleType");
    mt_mod.attr("NORMAL") = MeepleType::NORMAL;
    mt_mod.attr("ABBOT") = MeepleType::ABBOT;
    mt_mod.attr("FARMER") = MeepleType::FARMER;
    mt_mod.attr("BIG") = MeepleType::BIG;
    mt_mod.attr("BIG_FARMER") = MeepleType::BIG_FARMER;

    // ---- FarmerSide constants ----
    auto fs_mod = m.def_submodule("FarmerSide");
    fs_mod.attr("TLL") = FarmerSide::TLL;
    fs_mod.attr("TLT") = FarmerSide::TLT;
    fs_mod.attr("TRT") = FarmerSide::TRT;
    fs_mod.attr("TRR") = FarmerSide::TRR;
    fs_mod.attr("BRR") = FarmerSide::BRR;
    fs_mod.attr("BRB") = FarmerSide::BRB;
    fs_mod.attr("BLB") = FarmerSide::BLB;
    fs_mod.attr("BLL") = FarmerSide::BLL;
    fs_mod.def("get_side", &FarmerSide::get_side);

    // ---- GamePhase enum ----
    py::enum_<GamePhase>(m, "GamePhase")
        .value("TILES", GamePhase::TILES)
        .value("MEEPLES", GamePhase::MEEPLES)
        .def("to_json", [](GamePhase p) { return game_phase_to_string(p); })
        .def("__str__", [](GamePhase p) { return game_phase_to_string(p); })
        .def_property_readonly("value", [](GamePhase p) { return game_phase_to_string(p); });

    // ---- TerrainType enum ----
    py::enum_<TerrainType>(m, "TerrainType")
        .value("CITY", TerrainType::CITY)
        .value("GRASS", TerrainType::GRASS)
        .value("ROAD", TerrainType::ROAD)
        .value("CHAPEL", TerrainType::CHAPEL)
        .value("FLOWERS", TerrainType::FLOWERS)
        .value("UNPLAYABLE", TerrainType::UNPLAYABLE)
        .def("to_json", [](TerrainType t) { return terrain_type_to_string(t); })
        .def("__str__", [](TerrainType t) { return terrain_type_to_string(t); })
        .def_property_readonly("value", [](TerrainType t) { return terrain_type_to_string(t); });

    // ---- Rotation enum ----
    py::enum_<Rotation>(m, "Rotation")
        .value("CLOCKWISE", Rotation::CLOCKWISE)
        .value("COUNTER_CLOCKWISE", Rotation::COUNTER_CLOCKWISE)
        .value("NONE", Rotation::NONE)
        .def("to_json", [](Rotation r) { return rotation_to_string(r); })
        .def("__str__", [](Rotation r) { return rotation_to_string(r); })
        .def_property_readonly("value", [](Rotation r) { return rotation_to_string(r); });

    // ---- SupplementaryRule enum ----
    py::enum_<SupplementaryRule>(m, "SupplementaryRule")
        .value("FARMERS", SupplementaryRule::FARMERS)
        .value("ABBOTS", SupplementaryRule::ABBOTS)
        .value("NORMAL_MEEPLES_CAN_USE_FLOWERS", SupplementaryRule::NORMAL_MEEPLES_CAN_USE_FLOWERS)
        .def("to_json", [](SupplementaryRule r) { return supplementary_rule_to_string(r); })
        .def("__str__", [](SupplementaryRule r) { return supplementary_rule_to_string(r); })
        .def_property_readonly("value", [](SupplementaryRule r) { return supplementary_rule_to_string(r); });

    // ---- TileSet enum ----
    py::enum_<TileSet>(m, "TileSet")
        .value("BASE", TileSet::BASE)
        .value("THE_RIVER", TileSet::THE_RIVER)
        .value("INNS_AND_CATHEDRALS", TileSet::INNS_AND_CATHEDRALS)
        .def("to_json", [](TileSet t) { return tile_set_to_string(t); })
        .def("__str__", [](TileSet t) { return tile_set_to_string(t); })
        .def_property_readonly("value", [](TileSet t) { return tile_set_to_string(t); });

    // ---- Coordinate ----
    py::class_<Coordinate>(m, "Coordinate")
        .def(py::init<int, int>(), py::arg("row"), py::arg("column"))
        .def_readwrite("row", &Coordinate::row)
        .def_readwrite("column", &Coordinate::column)
        .def("__eq__", &Coordinate::operator==)
        .def("__hash__", [](const Coordinate& c) { return std::hash<Coordinate>{}(c); })
        .def("__str__", &Coordinate::to_string)
        .def("__repr__", &Coordinate::to_string)
        .def("to_json", &Coordinate::to_json)
        .def_static("from_json", &Coordinate::from_json);

    // ---- CoordinateWithSide ----
    py::class_<CoordinateWithSide>(m, "CoordinateWithSide")
        .def(py::init<Coordinate, int>(), py::arg("coordinate"), py::arg("side"))
        .def_readwrite("coordinate", &CoordinateWithSide::coordinate)
        .def_readwrite("side", &CoordinateWithSide::side)
        .def("__eq__", &CoordinateWithSide::operator==)
        .def("__hash__", [](const CoordinateWithSide& c) { return std::hash<CoordinateWithSide>{}(c); })
        .def("__str__", &CoordinateWithSide::to_string)
        .def("__repr__", &CoordinateWithSide::to_string)
        .def("to_json", [](const CoordinateWithSide& c) {
            py::list result;
            result.append(c.coordinate.to_json());
            result.append(c.side);
            return result;
        })
        .def_static("from_json", [](const py::list& data) {
            auto coord_data = data[0].cast<std::vector<int>>();
            return CoordinateWithSide(Coordinate::from_json(coord_data), data[1].cast<int>());
        });

    // ---- CoordinateWithFarmerSide ----
    py::class_<CoordinateWithFarmerSide>(m, "CoordinateWithFarmerSide")
        .def(py::init<Coordinate, int>())
        .def_readwrite("coordinate", &CoordinateWithFarmerSide::coordinate)
        .def_readwrite("farmer_side", &CoordinateWithFarmerSide::farmer_side)
        .def("__eq__", &CoordinateWithFarmerSide::operator==)
        .def("__hash__", [](const CoordinateWithFarmerSide& c) {
            return std::hash<CoordinateWithFarmerSide>{}(c);
        })
        .def("__str__", &CoordinateWithFarmerSide::to_string)
        .def("__repr__", &CoordinateWithFarmerSide::to_string);

    // ---- Connection ----
    py::class_<Connection>(m, "Connection")
        .def(py::init<int, int>(), py::arg("a"), py::arg("b"))
        .def_readwrite("a", &Connection::a)
        .def_readwrite("b", &Connection::b)
        .def("__eq__", &Connection::operator==)
        .def("__hash__", [](const Connection& c) { return std::hash<Connection>{}(c); });

    // ---- FarmerConnection ----
    py::class_<FarmerConnection>(m, "FarmerConnection")
        .def(py::init<std::vector<int>, std::vector<int>, std::vector<int>>(),
             py::arg("farmer_positions"), py::arg("tile_connections") = std::vector<int>{},
             py::arg("city_sides") = std::vector<int>{})
        .def_readwrite("farmer_positions", &FarmerConnection::farmer_positions)
        .def_readwrite("tile_connections", &FarmerConnection::tile_connections)
        .def_readwrite("city_sides", &FarmerConnection::city_sides)
        .def("__eq__", &FarmerConnection::operator==)
        .def("__hash__", [](const FarmerConnection& fc) { return std::hash<FarmerConnection>{}(fc); });

    // ---- FarmerConnectionWithCoordinate ----
    py::class_<FarmerConnectionWithCoordinate>(m, "FarmerConnectionWithCoordinate")
        .def(py::init<FarmerConnection, Coordinate>())
        .def_readwrite("farmer_connection", &FarmerConnectionWithCoordinate::farmer_connection)
        .def_readwrite("coordinate", &FarmerConnectionWithCoordinate::coordinate)
        .def("__eq__", &FarmerConnectionWithCoordinate::operator==);

    // ---- Tile ----
    py::class_<Tile>(m, "Tile")
        .def(py::init<>())
        .def_readwrite("description", &Tile::description)
        .def_readwrite("turns", &Tile::turns)
        .def_readwrite("road", &Tile::road)
        .def_readwrite("river", &Tile::river)
        .def_readwrite("city", &Tile::city)
        .def_readwrite("grass", &Tile::grass)
        .def_readwrite("farms", &Tile::farms)
        .def_readwrite("shield", &Tile::shield)
        .def_readwrite("chapel", &Tile::chapel)
        .def_readwrite("flowers", &Tile::flowers)
        .def_readwrite("inn", &Tile::inn)
        .def_readwrite("cathedral", &Tile::cathedral)
        .def_readwrite("unplayable_sides", &Tile::unplayable_sides)
        .def_readwrite("image", &Tile::image)
        .def("get_type", &Tile::get_type)
        .def("get_road_ends", &Tile::get_road_ends)
        .def("get_river_ends", &Tile::get_river_ends)
        .def("get_city_sides", &Tile::get_city_sides)
        .def("has_river", &Tile::has_river)
        .def("turn", &Tile::turn)
        .def("__eq__", &Tile::operator==)
        .def("__hash__", [](const Tile& t) { return std::hash<Tile>{}(t); })
        .def("__str__", [](const Tile& t) { return t.description; });

    // ---- MeeplePosition ----
    py::class_<MeeplePosition>(m, "MeeplePosition")
        .def(py::init<int, CoordinateWithSide>(), py::arg("meeple_type"), py::arg("coordinate_with_side"))
        .def_readwrite("meeple_type", &MeeplePosition::meeple_type)
        .def_readwrite("coordinate_with_side", &MeeplePosition::coordinate_with_side)
        .def("__eq__", &MeeplePosition::operator==)
        .def("__hash__", [](const MeeplePosition& mp) { return std::hash<MeeplePosition>{}(mp); })
        .def("to_json", [](const MeeplePosition& mp) {
            py::list result;
            result.append(mp.meeple_type);
            py::list cws;
            cws.append(mp.coordinate_with_side.coordinate.to_json());
            cws.append(mp.coordinate_with_side.side);
            result.append(cws);
            return result;
        })
        .def_static("from_json", [](const py::list& data) {
            int mt = data[0].cast<int>();
            auto cws_data = data[1].cast<py::list>();
            auto coord_data = cws_data[0].cast<std::vector<int>>();
            int side = cws_data[1].cast<int>();
            return MeeplePosition(mt, CoordinateWithSide(Coordinate::from_json(coord_data), side));
        });

    // ---- City ----
    py::class_<City>(m, "City")
        .def(py::init<>())
        .def_readonly("city_positions", &City::city_positions)
        .def_readonly("finished", &City::finished)
        .def("__eq__", &City::operator==)
        .def("__hash__", [](const City& c) { return std::hash<City>{}(c); });

    // ---- Road ----
    py::class_<Road>(m, "Road")
        .def(py::init<>())
        .def_readonly("road_positions", &Road::road_positions)
        .def_readonly("finished", &Road::finished)
        .def("__eq__", &Road::operator==)
        .def("__hash__", [](const Road& r) { return std::hash<Road>{}(r); });

    // ---- Farm ----
    py::class_<Farm>(m, "Farm")
        .def(py::init<>())
        .def_readonly("farmer_connections_with_coordinate", &Farm::farmer_connections_with_coordinate);

    // ---- PlayingPosition ----
    py::class_<PlayingPosition>(m, "PlayingPosition")
        .def(py::init<Coordinate, int>())
        .def_readwrite("coordinate", &PlayingPosition::coordinate)
        .def_readwrite("turns", &PlayingPosition::turns);

    // ---- PyAction wrapper ----
    py::class_<PyAction>(m, "Action")
        .def(py::init<>())
        .def("get_idx", &PyAction::get_idx)
        .def("__eq__", &PyAction::operator==)
        .def("__ne__", &PyAction::operator!=)
        .def("__hash__", [](const PyAction& a) { return std::hash<PyAction>{}(a); })
        .def("__str__", &PyAction::to_string)
        .def("__repr__", &PyAction::to_string)
        .def_property_readonly("_idx", &PyAction::get_idx);

    // ---- TileAction (Python-facing) ----
    py::class_<TileAction>(m, "TileAction")
        .def(py::init<Tile, Coordinate, int>(), py::arg("tile"), py::arg("coordinate"), py::arg("tile_rotations"))
        .def_readwrite("tile", &TileAction::tile)
        .def_readwrite("coordinate", &TileAction::coordinate)
        .def_readwrite("tile_rotations", &TileAction::tile_rotations)
        .def("get_idx", &TileAction::get_idx)
        .def("__eq__", [](const TileAction& a, const TileAction& b) { return a.get_idx() == b.get_idx(); })
        .def("__hash__", [](const TileAction& a) { return std::hash<int>{}(a.get_idx()); })
        .def("__str__", &TileAction::to_string)
        .def("__repr__", &TileAction::to_string)
        .def_property_readonly("_idx", &TileAction::get_idx)
        .def("to_json", [](const TileAction& ta) {
            py::list result;
            py::dict tile_json;
            tile_json["description"] = ta.tile.description;
            tile_json["turns"] = ta.tile.turns;
            result.append(tile_json);
            result.append(ta.coordinate.to_json());
            result.append(ta.tile_rotations);
            return result;
        })
        .def_static("from_json", [](const py::list& data) {
            auto& registry = get_global_tile_registry();
            auto tile_data = data[0].cast<py::dict>();
            std::string desc = tile_data["description"].cast<std::string>();
            int turns = tile_data["turns"].cast<int>();
            Tile t = registry.tile_from_json(desc, turns);
            auto coord_data = data[1].cast<std::vector<int>>();
            int rotations = data[2].cast<int>();
            return TileAction(std::move(t), Coordinate::from_json(coord_data), rotations);
        });

    // ---- MeepleAction (Python-facing) ----
    py::class_<MeepleAction>(m, "MeepleAction")
        .def(py::init<int, CoordinateWithSide, bool>(),
             py::arg("meeple_type"), py::arg("coordinate_with_side"), py::arg("remove") = false)
        .def_readwrite("meeple_type", &MeepleAction::meeple_type)
        .def_readwrite("coordinate_with_side", &MeepleAction::coordinate_with_side)
        .def_readwrite("remove", &MeepleAction::remove)
        .def("get_idx", &MeepleAction::get_idx)
        .def("__eq__", [](const MeepleAction& a, const MeepleAction& b) { return a.get_idx() == b.get_idx(); })
        .def("__hash__", [](const MeepleAction& a) { return std::hash<int>{}(a.get_idx()); })
        .def("__str__", &MeepleAction::to_string)
        .def("__repr__", &MeepleAction::to_string)
        .def_property_readonly("_idx", &MeepleAction::get_idx)
        .def("to_json", [](const MeepleAction& ma) {
            py::list result;
            result.append(ma.meeple_type);
            py::list cws;
            cws.append(ma.coordinate_with_side.coordinate.to_json());
            cws.append(ma.coordinate_with_side.side);
            result.append(cws);
            result.append(ma.remove);
            return result;
        })
        .def_static("from_json", [](const py::list& data) {
            int mt = data[0].cast<int>();
            auto cws_data = data[1].cast<py::list>();
            auto coord_data = cws_data[0].cast<std::vector<int>>();
            int side = cws_data[1].cast<int>();
            bool remove = data[2].cast<bool>();
            return MeepleAction(mt, CoordinateWithSide(Coordinate::from_json(coord_data), side), remove);
        });

    // ---- PassAction (Python-facing) ----
    py::class_<PassAction>(m, "PassAction")
        .def(py::init<>())
        .def("get_idx", &PassAction::get_idx)
        .def("__eq__", [](const PassAction& a, const PassAction& b) { return true; })
        .def("__hash__", [](const PassAction& a) { return std::hash<int>{}(a.get_idx()); })
        .def("__str__", &PassAction::to_string)
        .def("__repr__", &PassAction::to_string)
        .def_property_readonly("_idx", &PassAction::get_idx);

    // ---- TileRegistry ----
    py::class_<TileRegistry>(m, "TileRegistry")
        .def(py::init<>())
        .def_readwrite("tile_dict", &TileRegistry::tile_dict)
        .def("add_deck", &TileRegistry::add_deck)
        .def("tile_to_json", [](const TileRegistry& tr, const Tile& t) {
            py::dict d;
            d["description"] = t.description;
            d["turns"] = t.turns;
            return d;
        })
        .def("tile_from_json", [](const TileRegistry& tr, const py::dict& data) {
            std::string desc = data["description"].cast<std::string>();
            int turns = data["turns"].cast<int>();
            return tr.tile_from_json(desc, turns);
        });

    // ---- CarcassonneGameState ----
    py::class_<CarcassonneGameState>(m, "CarcassonneGameState")
        .def(py::init<>())
        .def(py::init<std::vector<TileSet>, std::vector<SupplementaryRule>, int, Coordinate>(),
             py::arg("tile_sets") = std::vector<TileSet>{TileSet::BASE, TileSet::INNS_AND_CATHEDRALS},
             py::arg("supplementary_rules") = std::vector<SupplementaryRule>{SupplementaryRule::FARMERS, SupplementaryRule::ABBOTS},
             py::arg("players") = 2,
             py::arg("starting_position") = Coordinate(0, 0))
        .def_readwrite("deck", &CarcassonneGameState::deck)
        .def_readwrite("supplementary_rules", &CarcassonneGameState::supplementary_rules)
        .def_readwrite("starting_position", &CarcassonneGameState::starting_position)
        .def_readwrite("players", &CarcassonneGameState::players)
        .def_readwrite("meeples", &CarcassonneGameState::meeples)
        .def_readwrite("abbots", &CarcassonneGameState::abbots)
        .def_readwrite("big_meeples", &CarcassonneGameState::big_meeples)
        .def_readwrite("placed_meeples", &CarcassonneGameState::placed_meeples)
        .def_readwrite("scores", &CarcassonneGameState::scores)
        .def_readwrite("current_player", &CarcassonneGameState::current_player)
        .def_readwrite("phase", &CarcassonneGameState::phase)
        .def_readwrite("last_river_rotation", &CarcassonneGameState::last_river_rotation)
        .def_property("next_tile",
            [](const CarcassonneGameState& s) -> py::object {
                if (s.next_tile.has_value()) return py::cast(*s.next_tile);
                return py::none();
            },
            [](CarcassonneGameState& s, py::object val) {
                if (val.is_none()) s.next_tile = std::nullopt;
                else s.next_tile = val.cast<Tile>();
            })
        .def_property("last_tile_action",
            [](const CarcassonneGameState& s) -> py::object {
                if (s.last_tile_action.has_value()) return py::cast(*s.last_tile_action);
                return py::none();
            },
            [](CarcassonneGameState& s, py::object val) {
                if (val.is_none()) s.last_tile_action = std::nullopt;
                else s.last_tile_action = val.cast<TileAction>();
            })
        .def_property("board",
            [](const CarcassonneGameState& s) {
                py::dict d;
                for (auto& [k, v] : s.board) {
                    d[py::make_tuple(k.first, k.second)] = v;
                }
                return d;
            },
            [](CarcassonneGameState& s, const py::dict& d) {
                s.board.clear();
                for (auto& [k, v] : d) {
                    auto tup = k.cast<py::tuple>();
                    int r = tup[0].cast<int>();
                    int c = tup[1].cast<int>();
                    s.board[{r, c}] = v.cast<Tile>();
                }
            })
        .def("get_tile", [](const CarcassonneGameState& s, int row, int col) -> py::object {
            const Tile* t = s.get_tile(row, col);
            if (t) return py::cast(*t);
            return py::none();
        })
        .def("empty_board", &CarcassonneGameState::empty_board)
        .def("is_terminated", &CarcassonneGameState::is_terminated)
        .def("simple_copy", &CarcassonneGameState::simple_copy)
        .def("to_json", [](const CarcassonneGameState& s) {
            py::dict data;

            py::list deck_json;
            for (auto& t : s.deck) {
                py::dict td;
                td["description"] = t.description;
                td["turns"] = t.turns;
                deck_json.append(td);
            }
            data["deck"] = deck_json;

            py::list rules_json;
            for (auto& r : s.supplementary_rules)
                rules_json.append(supplementary_rule_to_string(r));
            data["supplementary_rules"] = rules_json;

            py::dict board_json;
            for (auto& [pos, tile] : s.board) {
                std::string key = std::to_string(pos.first) + "," + std::to_string(pos.second);
                py::dict td;
                td["description"] = tile.description;
                td["turns"] = tile.turns;
                board_json[py::cast(key)] = td;
            }
            data["board"] = board_json;

            data["starting_position"] = s.starting_position.to_json();

            if (s.next_tile.has_value()) {
                py::dict nt;
                nt["description"] = s.next_tile->description;
                nt["turns"] = s.next_tile->turns;
                data["next_tile"] = nt;
            } else {
                data["next_tile"] = py::none();
            }

            data["players"] = s.players;
            data["meeples"] = s.meeples;
            data["abbots"] = s.abbots;
            data["big_meeples"] = s.big_meeples;

            py::list pm_json;
            for (auto& player_meeples : s.placed_meeples) {
                py::list player_list;
                for (auto& mp : player_meeples) {
                    py::list mp_json;
                    mp_json.append(mp.meeple_type);
                    py::list cws_json;
                    cws_json.append(mp.coordinate_with_side.coordinate.to_json());
                    cws_json.append(mp.coordinate_with_side.side);
                    mp_json.append(cws_json);
                    player_list.append(mp_json);
                }
                pm_json.append(player_list);
            }
            data["placed_meeples"] = pm_json;

            data["scores"] = s.scores;
            data["current_player"] = s.current_player;
            data["phase"] = game_phase_to_string(s.phase);

            if (s.last_tile_action.has_value()) {
                py::list lta;
                py::dict td;
                td["description"] = s.last_tile_action->tile.description;
                td["turns"] = s.last_tile_action->tile.turns;
                lta.append(td);
                lta.append(s.last_tile_action->coordinate.to_json());
                lta.append(s.last_tile_action->tile_rotations);
                data["last_tile_action"] = lta;
            } else {
                data["last_tile_action"] = py::none();
            }

            data["last_river_rotation"] = rotation_to_string(s.last_river_rotation);

            return data;
        })
        .def_static("from_json", [](const py::dict& data) {
            auto& registry = get_global_tile_registry();
            CarcassonneGameState s;

            auto deck_data = data["deck"].cast<py::list>();
            for (auto& item : deck_data) {
                auto td = item.cast<py::dict>();
                s.deck.push_back(registry.tile_from_json(
                    td["description"].cast<std::string>(),
                    td["turns"].cast<int>()));
            }

            auto rules_data = data["supplementary_rules"].cast<py::list>();
            for (auto& r : rules_data) {
                s.supplementary_rules.push_back(supplementary_rule_from_string(r.cast<std::string>()));
            }

            auto board_data = data["board"].cast<py::dict>();
            for (auto& [key, val] : board_data) {
                std::string k = key.cast<std::string>();
                auto comma = k.find(',');
                int r = std::stoi(k.substr(0, comma));
                int c = std::stoi(k.substr(comma + 1));
                auto td = val.cast<py::dict>();
                s.board[{r, c}] = registry.tile_from_json(
                    td["description"].cast<std::string>(),
                    td["turns"].cast<int>());
            }

            auto sp = data["starting_position"].cast<std::vector<int>>();
            s.starting_position = Coordinate::from_json(sp);

            if (!data["next_tile"].is_none()) {
                auto nt = data["next_tile"].cast<py::dict>();
                s.next_tile = registry.tile_from_json(
                    nt["description"].cast<std::string>(),
                    nt["turns"].cast<int>());
            }

            s.players = data["players"].cast<int>();
            s.meeples = data["meeples"].cast<std::vector<int>>();
            s.abbots = data["abbots"].cast<std::vector<int>>();
            s.big_meeples = data["big_meeples"].cast<std::vector<int>>();

            auto pm_data = data["placed_meeples"].cast<py::list>();
            for (auto& player_data : pm_data) {
                std::vector<MeeplePosition> player_meeples;
                for (auto& mp_data : player_data.cast<py::list>()) {
                    auto mp_list = mp_data.cast<py::list>();
                    int mt = mp_list[0].cast<int>();
                    auto cws_data = mp_list[1].cast<py::list>();
                    auto coord_data = cws_data[0].cast<std::vector<int>>();
                    int side = cws_data[1].cast<int>();
                    player_meeples.push_back(MeeplePosition(mt,
                        CoordinateWithSide(Coordinate::from_json(coord_data), side)));
                }
                s.placed_meeples.push_back(std::move(player_meeples));
            }

            s.scores = data["scores"].cast<std::vector<int>>();
            s.current_player = data["current_player"].cast<int>();
            s.phase = game_phase_from_string(data["phase"].cast<std::string>());

            if (!data["last_tile_action"].is_none()) {
                auto lta_data = data["last_tile_action"].cast<py::list>();
                auto td = lta_data[0].cast<py::dict>();
                Tile t = registry.tile_from_json(
                    td["description"].cast<std::string>(),
                    td["turns"].cast<int>());
                auto coord = Coordinate::from_json(lta_data[1].cast<std::vector<int>>());
                int rotations = lta_data[2].cast<int>();
                s.last_tile_action = TileAction(std::move(t), coord, rotations);
            }

            s.last_river_rotation = rotation_from_string(
                data["last_river_rotation"].cast<std::string>());

            return s;
        });

    // ---- Utility classes ----
    py::class_<ActionUtil>(m, "ActionUtil")
        .def_static("get_possible_actions", [](const CarcassonneGameState& state) {
            auto actions = ActionUtil::get_possible_actions(state);
            py::list result;
            for (auto& a : actions) {
                if (a->type() == ActionType::TILE)
                    result.append(py::cast(*static_cast<TileAction*>(a.get())));
                else if (a->type() == ActionType::MEEPLE)
                    result.append(py::cast(*static_cast<MeepleAction*>(a.get())));
                else
                    result.append(py::cast(PassAction()));
            }
            return result;
        });

    py::class_<StateUpdater>(m, "StateUpdater")
        .def_static("apply_action", [](const CarcassonneGameState& state, py::object action_obj) {
            if (py::isinstance<TileAction>(action_obj)) {
                auto& ta = action_obj.cast<TileAction&>();
                return StateUpdater::apply_action(state, ta);
            } else if (py::isinstance<MeepleAction>(action_obj)) {
                auto& ma = action_obj.cast<MeepleAction&>();
                return StateUpdater::apply_action(state, ma);
            } else {
                PassAction pa;
                return StateUpdater::apply_action(state, pa);
            }
        });

    py::class_<TileFitter>(m, "TileFitter")
        .def_static("fits", [](const Tile& center,
                               py::object top, py::object right, py::object bottom, py::object left,
                               py::object game_state) {
            const Tile* t = top.is_none() ? nullptr : &top.cast<Tile&>();
            const Tile* r = right.is_none() ? nullptr : &right.cast<Tile&>();
            const Tile* b = bottom.is_none() ? nullptr : &bottom.cast<Tile&>();
            const Tile* l = left.is_none() ? nullptr : &left.cast<Tile&>();
            const CarcassonneGameState* gs = game_state.is_none() ? nullptr : &game_state.cast<CarcassonneGameState&>();
            return TileFitter::fits(center, t, r, b, l, gs);
        }, py::arg("center"), py::arg("top") = py::none(), py::arg("right") = py::none(),
           py::arg("bottom") = py::none(), py::arg("left") = py::none(),
           py::arg("game_state") = py::none());

    py::class_<CityUtil>(m, "CityUtil")
        .def_static("find_city", &CityUtil::find_city)
        .def_static("find_cities", &CityUtil::find_cities,
                     py::arg("state"), py::arg("coordinate"),
                     py::arg("sides") = std::vector<int>{Side::TOP, Side::RIGHT, Side::BOTTOM, Side::LEFT})
        .def_static("find_meeples", &CityUtil::find_meeples)
        .def_static("city_contains_meeples", &CityUtil::city_contains_meeples);

    py::class_<RoadUtil>(m, "RoadUtil")
        .def_static("find_road", &RoadUtil::find_road)
        .def_static("find_roads", &RoadUtil::find_roads)
        .def_static("find_meeples", &RoadUtil::find_meeples)
        .def_static("road_contains_meeples", &RoadUtil::road_contains_meeples);

    py::class_<FarmUtil>(m, "FarmUtil")
        .def_static("find_farm", &FarmUtil::find_farm)
        .def_static("find_farm_by_coordinate", &FarmUtil::find_farm_by_coordinate)
        .def_static("find_meeples", &FarmUtil::find_meeples)
        .def_static("has_meeples", &FarmUtil::has_meeples);

    py::class_<PointsCollector>(m, "PointsCollector")
        .def_static("count_city_points", &PointsCollector::count_city_points)
        .def_static("count_road_points", &PointsCollector::count_road_points)
        .def_static("chapel_or_flowers_points", &PointsCollector::chapel_or_flowers_points)
        .def_static("count_farm_points", &PointsCollector::count_farm_points);

    // ---- Top-level functions ----
    m.def("action_to_json", [](py::object action_obj) -> py::dict {
        py::dict result;
        if (py::isinstance<TileAction>(action_obj)) {
            auto& ta = action_obj.cast<TileAction&>();
            result["type"] = "TileAction";
            py::list data;
            py::dict td;
            td["description"] = ta.tile.description;
            td["turns"] = ta.tile.turns;
            data.append(td);
            data.append(ta.coordinate.to_json());
            data.append(ta.tile_rotations);
            result["data"] = data;
        } else if (py::isinstance<MeepleAction>(action_obj)) {
            auto& ma = action_obj.cast<MeepleAction&>();
            result["type"] = "MeepleAction";
            py::list data;
            data.append(ma.meeple_type);
            py::list cws;
            cws.append(ma.coordinate_with_side.coordinate.to_json());
            cws.append(ma.coordinate_with_side.side);
            data.append(cws);
            data.append(ma.remove);
            result["data"] = data;
        } else {
            result["type"] = "PassAction";
        }
        return result;
    });

    m.def("action_from_json", [](const py::dict& data) -> py::object {
        auto& registry = get_global_tile_registry();
        std::string type = data["type"].cast<std::string>();
        if (type == "TileAction") {
            auto d = data["data"].cast<py::list>();
            auto td = d[0].cast<py::dict>();
            Tile t = registry.tile_from_json(
                td["description"].cast<std::string>(),
                td["turns"].cast<int>());
            auto coord = Coordinate::from_json(d[1].cast<std::vector<int>>());
            int rot = d[2].cast<int>();
            return py::cast(TileAction(std::move(t), coord, rot));
        } else if (type == "MeepleAction") {
            auto d = data["data"].cast<py::list>();
            int mt = d[0].cast<int>();
            auto cws_data = d[1].cast<py::list>();
            auto coord_data = cws_data[0].cast<std::vector<int>>();
            int side = cws_data[1].cast<int>();
            bool remove = d[2].cast<bool>();
            return py::cast(MeepleAction(mt,
                CoordinateWithSide(Coordinate::from_json(coord_data), side), remove));
        } else {
            return py::cast(PassAction());
        }
    });

    // Expose tile dictionaries
    m.def("get_base_tiles", []() -> std::unordered_map<std::string, Tile> {
        return get_base_deck().tiles;
    });
    m.def("get_inns_and_cathedrals_tiles", []() -> std::unordered_map<std::string, Tile> {
        return get_inns_and_cathedrals_deck().tiles;
    });
    m.def("get_the_river_tiles", []() -> std::unordered_map<std::string, Tile> {
        return get_the_river_deck().tiles;
    });

    m.attr("base_tiles") = get_base_deck().tiles;
    m.attr("inns_and_cathedrals_tiles") = get_inns_and_cathedrals_deck().tiles;

    m.attr("tile_registry") = get_global_tile_registry();

    // ---- Graph feature construction ----
    m.def("build_graph_features", [](const CarcassonneGameState& state,
                                      const std::unordered_map<std::string, int>& tile_desc_to_idx,
                                      int num_tile_types) {
        auto gf = build_graph_features(state, tile_desc_to_idx, num_tile_types);
        py::dict result;
        result["num_nodes"] = gf.num_nodes;
        result["num_edges"] = gf.num_edges;
        result["last_placed_node_idx"] = gf.last_placed_node_idx;
        result["origin_coord"] = py::make_tuple(gf.origin_row, gf.origin_col);

        // x_gps as flat list (Python side will reshape)
        result["x_gps"] = std::move(gf.x_gps);
        result["edge_index"] = std::move(gf.edge_index);
        result["edge_attr"] = std::move(gf.edge_attr);
        return result;
    }, py::arg("state"), py::arg("tile_desc_to_idx"), py::arg("num_tile_types") = 50);

    m.attr("USING_CPP_ENGINE") = true;
}

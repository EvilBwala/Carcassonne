#include "carcassonne/tile_sets.h"
#include "carcassonne/types.h"

namespace carcassonne {

static Tile make_tile(const std::string& desc,
                      std::vector<Connection> road = {},
                      std::vector<Connection> river = {},
                      std::vector<std::vector<int>> city = {},
                      std::vector<int> grass = {},
                      std::vector<FarmerConnection> farms = {},
                      bool shield = false,
                      bool chapel = false,
                      bool flowers = false,
                      std::vector<int> inn = {},
                      bool cathedral = false,
                      std::vector<int> unplayable_sides = {},
                      const std::string& image = "") {
    Tile t;
    t.description = desc;
    t.turns = 0;
    t.road = std::move(road);
    t.river = std::move(river);
    t.city = std::move(city);
    t.grass = std::move(grass);
    t.farms = std::move(farms);
    t.shield = shield;
    t.chapel = chapel;
    t.flowers = flowers;
    t.inn = std::move(inn);
    t.cathedral = cathedral;
    t.unplayable_sides = std::move(unplayable_sides);
    t.image = image;
    return t;
}

namespace S = Side;
namespace FS = FarmerSide;

const TileDeck& get_base_deck() {
    static TileDeck deck = []() {
        TileDeck d;
        auto& T = d.tiles;

        T["chapel_with_road"] = make_tile("chapel_with_road",
            /*road*/ {{S::BOTTOM, S::CENTER}},
            /*river*/ {},
            /*city*/ {},
            /*grass*/ {S::LEFT, S::TOP, S::RIGHT},
            /*farms*/ {FarmerConnection(
                {S::TOP_LEFT, S::TOP_RIGHT, S::BOTTOM_LEFT, S::BOTTOM_RIGHT},
                {FS::TLL, FS::TLT, FS::TRT, FS::TRR, FS::BRR, FS::BRB, FS::BLB, FS::BLL}
            )},
            /*shield*/ false, /*chapel*/ true
        );

        T["chapel"] = make_tile("chapel",
            {}, {}, {},
            {S::TOP, S::RIGHT, S::BOTTOM, S::LEFT},
            {FarmerConnection(
                {S::TOP_LEFT, S::TOP_RIGHT, S::BOTTOM_LEFT, S::BOTTOM_RIGHT},
                {FS::TLL, FS::TLT, FS::TRT, FS::TRR, FS::BRR, FS::BRB, FS::BLB, FS::BLL}
            )},
            false, true
        );

        T["full_city_with_shield"] = make_tile("full_city_with_shield",
            {}, {}, {{S::TOP, S::RIGHT, S::BOTTOM, S::LEFT}},
            {}, {}, true
        );

        T["city_top_straight_road"] = make_tile("city_top_straight_road",
            {{S::LEFT, S::RIGHT}}, {}, {{S::TOP}},
            {S::BOTTOM},
            {
                FarmerConnection({S::TOP_LEFT, S::TOP_RIGHT},
                    {FS::TLL, FS::TLT, FS::TRT, FS::TRR}, {S::TOP}),
                FarmerConnection({S::BOTTOM_LEFT, S::BOTTOM_RIGHT},
                    {FS::BRR, FS::BRB, FS::BLB, FS::BLL})
            }
        );

        T["city_top"] = make_tile("city_top",
            {}, {}, {{S::TOP}},
            {S::RIGHT, S::BOTTOM, S::LEFT},
            {FarmerConnection(
                {S::TOP_LEFT, S::TOP_RIGHT, S::BOTTOM_LEFT, S::BOTTOM_RIGHT},
                {FS::TLL, FS::TRR, FS::BRR, FS::BRB, FS::BLB, FS::BLL},
                {S::TOP}
            )}
        );

        T["city_top_flowers"] = make_tile("city_top_flowers",
            {}, {}, {{S::TOP}},
            {S::RIGHT, S::BOTTOM, S::LEFT},
            {FarmerConnection(
                {S::TOP_LEFT, S::TOP_RIGHT, S::BOTTOM_LEFT, S::BOTTOM_RIGHT},
                {FS::TLL, FS::TRR, FS::BRR, FS::BRB, FS::BLB, FS::BLL},
                {S::TOP}
            )},
            false, false, true
        );

        T["city_narrow_shield"] = make_tile("city_narrow_shield",
            {}, {}, {{S::LEFT, S::RIGHT}},
            {S::TOP, S::BOTTOM},
            {
                FarmerConnection({S::TOP_LEFT, S::TOP_RIGHT},
                    {FS::TLT, FS::TRT}, {S::LEFT, S::RIGHT}),
                FarmerConnection({S::BOTTOM_LEFT, S::BOTTOM_RIGHT},
                    {FS::BRB, FS::BLB}, {S::LEFT, S::RIGHT})
            },
            true
        );

        T["city_narrow"] = make_tile("city_narrow",
            {}, {}, {{S::LEFT, S::RIGHT}},
            {S::TOP, S::BOTTOM},
            {
                FarmerConnection({S::TOP_LEFT, S::TOP_RIGHT},
                    {FS::TLT, FS::TRT}, {S::LEFT, S::RIGHT}),
                FarmerConnection({S::BOTTOM_LEFT, S::BOTTOM_RIGHT},
                    {FS::BRB, FS::BLB}, {S::LEFT, S::RIGHT})
            }
        );

        T["city_left_right"] = make_tile("city_left_right",
            {}, {}, {{S::LEFT}, {S::RIGHT}},
            {S::TOP, S::BOTTOM, S::CENTER},
            {FarmerConnection(
                {S::TOP_LEFT, S::TOP_RIGHT, S::BOTTOM_LEFT, S::BOTTOM_RIGHT},
                {FS::TLT, FS::TRT, FS::BRB, FS::BLB},
                {S::LEFT, S::RIGHT}
            )}
        );

        T["city_top_bottom_flowers"] = make_tile("city_top_bottom_flowers",
            {}, {}, {{S::TOP}, {S::BOTTOM}},
            {S::LEFT, S::RIGHT},
            {FarmerConnection(
                {S::TOP_LEFT, S::TOP_RIGHT, S::BOTTOM_LEFT, S::BOTTOM_RIGHT},
                {FS::TLL, FS::TRR, FS::BRR, FS::BLL},
                {S::TOP, S::BOTTOM}
            )},
            false, false, true
        );

        T["city_top_right"] = make_tile("city_top_right",
            {}, {}, {{S::TOP}, {S::RIGHT}},
            {S::LEFT, S::BOTTOM, S::CENTER},
            {FarmerConnection(
                {S::TOP_LEFT, S::TOP_RIGHT, S::BOTTOM_LEFT, S::BOTTOM_RIGHT},
                {FS::TLL, FS::BRB, FS::BLB, FS::BLL},
                {S::TOP, S::RIGHT}
            )}
        );

        T["city_top_left_flowers"] = make_tile("city_top_left_flowers",
            {}, {}, {{S::TOP}, {S::LEFT}},
            {S::BOTTOM, S::RIGHT},
            {FarmerConnection(
                {S::TOP_LEFT, S::TOP_RIGHT, S::BOTTOM_LEFT, S::BOTTOM_RIGHT},
                {FS::TRR, FS::BRR, FS::BRB, FS::BLB},
                {S::LEFT, S::TOP}
            )},
            false, false, true
        );

        T["city_top_road_bend_right"] = make_tile("city_top_road_bend_right",
            {{S::BOTTOM, S::RIGHT}}, {}, {{S::TOP}},
            {S::LEFT},
            {
                FarmerConnection({S::TOP_LEFT, S::TOP_RIGHT, S::BOTTOM_LEFT},
                    {FS::TLL, FS::TRR, FS::BLB, FS::BLL}, {S::TOP}),
                FarmerConnection({S::BOTTOM_RIGHT},
                    {FS::BRR, FS::BRB})
            }
        );

        T["city_top_road_bend_left"] = make_tile("city_top_road_bend_left",
            {{S::BOTTOM, S::LEFT}}, {}, {{S::TOP}},
            {S::RIGHT},
            {
                FarmerConnection({S::TOP_LEFT, S::TOP_RIGHT, S::BOTTOM_RIGHT},
                    {FS::TLL, FS::TRR, FS::BRB, FS::BRR}, {S::TOP}),
                FarmerConnection({S::BOTTOM_LEFT},
                    {FS::BLL, FS::BLB})
            }
        );

        T["city_top_crossroads"] = make_tile("city_top_crossroads",
            {{S::BOTTOM, S::CENTER}, {S::LEFT, S::CENTER}, {S::RIGHT, S::CENTER}},
            {}, {{S::TOP}}, {},
            {
                FarmerConnection({S::TOP_LEFT, S::TOP_RIGHT},
                    {FS::TLL, FS::TRR}, {S::TOP}),
                FarmerConnection({S::BOTTOM_LEFT},
                    {FS::BLL, FS::BLB}),
                FarmerConnection({S::BOTTOM_RIGHT},
                    {FS::BRB, FS::BRR})
            }
        );

        T["city_diagonal_top_right_shield"] = make_tile("city_diagonal_top_right_shield",
            {}, {}, {{S::TOP, S::RIGHT}},
            {S::LEFT, S::BOTTOM},
            {FarmerConnection(
                {S::TOP_LEFT, S::BOTTOM_LEFT, S::BOTTOM_RIGHT},
                {FS::TLL, FS::BLB, FS::BLL, FS::BRB},
                {S::TOP, S::RIGHT}
            )},
            true
        );

        T["city_diagonal_top_right_shield_flowers"] = make_tile("city_diagonal_top_right_shield_flowers",
            {}, {}, {{S::TOP, S::RIGHT}},
            {S::LEFT, S::BOTTOM},
            {FarmerConnection(
                {S::TOP_LEFT, S::BOTTOM_LEFT, S::BOTTOM_RIGHT},
                {FS::TLL, FS::BLB, FS::BLL, FS::BRB},
                {S::TOP, S::RIGHT}
            )},
            true, false, true
        );

        T["city_diagonal_top_right"] = make_tile("city_diagonal_top_right",
            {}, {}, {{S::TOP, S::RIGHT}},
            {S::LEFT, S::BOTTOM},
            {FarmerConnection(
                {S::TOP_LEFT, S::BOTTOM_LEFT, S::BOTTOM_RIGHT},
                {FS::TLL, FS::BLB, FS::BLL, FS::BRB},
                {S::TOP, S::RIGHT}
            )}
        );

        T["city_diagonal_top_right_flowers"] = make_tile("city_diagonal_top_right_flowers",
            {}, {}, {{S::TOP, S::RIGHT}},
            {S::LEFT, S::BOTTOM},
            {FarmerConnection(
                {S::TOP_LEFT, S::BOTTOM_LEFT, S::BOTTOM_RIGHT},
                {FS::TLL, FS::BLB, FS::BLL, FS::BRB},
                {S::TOP, S::RIGHT}
            )},
            false, false, true
        );

        T["city_diagonal_top_left_shield_road"] = make_tile("city_diagonal_top_left_shield_road",
            {{S::BOTTOM, S::RIGHT}}, {}, {{S::TOP, S::LEFT}}, {},
            {
                FarmerConnection({S::BOTTOM_LEFT, S::TOP_RIGHT},
                    {FS::BLB, FS::TRR}, {S::TOP, S::LEFT}),
                FarmerConnection({S::BOTTOM_RIGHT},
                    {FS::BRR, FS::BRB})
            },
            true
        );

        T["city_diagonal_top_left_road"] = make_tile("city_diagonal_top_left_road",
            {{S::BOTTOM, S::RIGHT}}, {}, {{S::TOP, S::LEFT}}, {},
            {
                FarmerConnection({S::BOTTOM_LEFT, S::TOP_RIGHT},
                    {FS::BLB, FS::TRR}, {S::TOP, S::LEFT}),
                FarmerConnection({S::BOTTOM_RIGHT},
                    {FS::BRR, FS::BRB})
            }
        );

        T["city_bottom_grass_shield"] = make_tile("city_bottom_grass_shield",
            {}, {}, {{S::TOP, S::LEFT, S::RIGHT}},
            {S::BOTTOM},
            {FarmerConnection(
                {S::BOTTOM_LEFT, S::BOTTOM_RIGHT},
                {FS::BLB, FS::BRB},
                {S::TOP, S::LEFT, S::RIGHT}
            )},
            true
        );

        T["city_bottom_grass"] = make_tile("city_bottom_grass",
            {}, {}, {{S::TOP, S::LEFT, S::RIGHT}},
            {S::BOTTOM},
            {FarmerConnection(
                {S::BOTTOM_LEFT, S::BOTTOM_RIGHT},
                {FS::BLB, FS::BRB},
                {S::TOP, S::LEFT, S::RIGHT}
            )}
        );

        T["city_bottom_grass_flowers"] = make_tile("city_bottom_grass_flowers",
            {}, {}, {{S::TOP, S::LEFT, S::RIGHT}},
            {S::BOTTOM},
            {FarmerConnection(
                {S::BOTTOM_LEFT, S::BOTTOM_RIGHT},
                {FS::BLB, FS::BRB},
                {S::TOP, S::LEFT, S::RIGHT}
            )},
            false, false, true
        );

        T["city_bottom_road_shield"] = make_tile("city_bottom_road_shield",
            {{S::BOTTOM, S::CENTER}}, {}, {{S::TOP, S::LEFT, S::RIGHT}}, {},
            {
                FarmerConnection({S::BOTTOM_LEFT},
                    {FS::BLB}, {S::TOP, S::LEFT, S::RIGHT}),
                FarmerConnection({S::BOTTOM_RIGHT},
                    {FS::BRB}, {S::TOP, S::LEFT, S::RIGHT})
            },
            true
        );

        T["city_bottom_road"] = make_tile("city_bottom_road",
            {{S::BOTTOM, S::CENTER}}, {}, {{S::TOP, S::LEFT, S::RIGHT}}, {},
            {
                FarmerConnection({S::BOTTOM_LEFT},
                    {FS::BLB}, {S::TOP, S::LEFT, S::RIGHT}),
                FarmerConnection({S::BOTTOM_RIGHT},
                    {FS::BRB}, {S::TOP, S::LEFT, S::RIGHT})
            }
        );

        T["straight_road"] = make_tile("straight_road",
            {{S::BOTTOM, S::TOP}}, {}, {},
            {S::LEFT, S::RIGHT},
            {
                FarmerConnection({S::TOP_LEFT, S::BOTTOM_LEFT},
                    {FS::TLL, FS::TLT, FS::BLB, FS::BLL}),
                FarmerConnection({S::TOP_RIGHT, S::BOTTOM_RIGHT},
                    {FS::TRR, FS::TRT, FS::BRR, FS::BRB})
            }
        );

        T["straight_road_flowers"] = make_tile("straight_road_flowers",
            {{S::BOTTOM, S::TOP}}, {}, {},
            {S::LEFT, S::RIGHT},
            {
                FarmerConnection({S::TOP_LEFT, S::BOTTOM_LEFT},
                    {FS::TLL, FS::TLT, FS::BLB, FS::BLL}),
                FarmerConnection({S::TOP_RIGHT, S::BOTTOM_RIGHT},
                    {FS::TRR, FS::TRT, FS::BRR, FS::BRB})
            },
            false, false, true
        );

        T["bent_road"] = make_tile("bent_road",
            {{S::LEFT, S::BOTTOM}}, {}, {},
            {S::TOP, S::RIGHT},
            {
                FarmerConnection({S::BOTTOM_LEFT},
                    {FS::BLB, FS::BLL}),
                FarmerConnection({S::TOP_LEFT, S::TOP_RIGHT, S::BOTTOM_RIGHT},
                    {FS::TLL, FS::TLT, FS::TRR, FS::TRT, FS::BRB, FS::BRR})
            }
        );

        T["bent_road_flowers"] = make_tile("bent_road_flowers",
            {{S::LEFT, S::BOTTOM}}, {}, {},
            {S::TOP, S::RIGHT},
            {
                FarmerConnection({S::BOTTOM_LEFT},
                    {FS::BLB, FS::BLL}),
                FarmerConnection({S::TOP_LEFT, S::TOP_RIGHT, S::BOTTOM_RIGHT},
                    {FS::TLL, FS::TLT, FS::TRR, FS::TRT, FS::BRB, FS::BRR})
            },
            false, false, true
        );

        T["three_split_road"] = make_tile("three_split_road",
            {{S::BOTTOM, S::CENTER}, {S::LEFT, S::CENTER}, {S::RIGHT, S::CENTER}},
            {}, {},
            {S::TOP},
            {
                FarmerConnection({S::BOTTOM_LEFT},
                    {FS::BLB, FS::BLL}),
                FarmerConnection({S::BOTTOM_RIGHT},
                    {FS::BRB, FS::BRR}),
                FarmerConnection({S::TOP_LEFT, S::TOP_RIGHT},
                    {FS::TLL, FS::TLT, FS::TRR, FS::TRT})
            }
        );

        T["crossroads"] = make_tile("crossroads",
            {{S::BOTTOM, S::CENTER}, {S::LEFT, S::CENTER}, {S::RIGHT, S::CENTER}, {S::TOP, S::CENTER}},
            {}, {}, {},
            {
                FarmerConnection({S::BOTTOM_LEFT}, {FS::BLB, FS::BLL}),
                FarmerConnection({S::BOTTOM_RIGHT}, {FS::BRB, FS::BRR}),
                FarmerConnection({S::TOP_LEFT}, {FS::TLL, FS::TLT}),
                FarmerConnection({S::TOP_RIGHT}, {FS::TRR, FS::TRT})
            }
        );

        d.tile_counts = {
            {"chapel_with_road", 2}, {"chapel", 4}, {"full_city_with_shield", 1},
            {"city_top_straight_road", 4}, {"city_top", 4}, {"city_top_flowers", 1},
            {"city_narrow_shield", 2}, {"city_narrow", 1}, {"city_left_right", 2},
            {"city_top_bottom_flowers", 1}, {"city_top_right", 1}, {"city_top_left_flowers", 1},
            {"city_top_road_bend_right", 3}, {"city_top_road_bend_left", 3},
            {"city_top_crossroads", 3}, {"city_diagonal_top_right_shield", 1},
            {"city_diagonal_top_right_shield_flowers", 1}, {"city_diagonal_top_right", 2},
            {"city_diagonal_top_right_flowers", 1}, {"city_diagonal_top_left_shield_road", 2},
            {"city_diagonal_top_left_road", 3}, {"city_bottom_grass_shield", 1},
            {"city_bottom_grass", 2}, {"city_bottom_grass_flowers", 1},
            {"city_bottom_road_shield", 2}, {"city_bottom_road", 1},
            {"straight_road", 7}, {"straight_road_flowers", 1},
            {"bent_road", 8}, {"bent_road_flowers", 1},
            {"three_split_road", 4}, {"crossroads", 1}
        };

        return d;
    }();
    return deck;
}

} // namespace carcassonne

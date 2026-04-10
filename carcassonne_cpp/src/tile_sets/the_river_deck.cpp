#include "carcassonne/tile_sets.h"
#include "carcassonne/types.h"

namespace carcassonne {

namespace S = Side;
namespace FS = FarmerSide;

const TileDeck& get_the_river_deck() {
    static TileDeck deck = []() {
        TileDeck d;
        auto& T = d.tiles;

        // river_start
        {
            Tile t;
            t.description = "river_start";
            t.river = {{S::CENTER, S::BOTTOM}};
            t.grass = {S::LEFT, S::RIGHT, S::TOP};
            t.farms = {FarmerConnection(
                {S::TOP_LEFT, S::TOP_RIGHT, S::BOTTOM_LEFT, S::BOTTOM_RIGHT},
                {FS::TLL, FS::TLT, FS::TRT, FS::TRR, FS::BRR, FS::BRB, FS::BLB, FS::BLL}
            )};
            T["river_start"] = std::move(t);
        }

        // river_city_with_road
        {
            Tile t;
            t.description = "river_city_with_road";
            t.river = {{S::LEFT, S::RIGHT}};
            t.road = {{S::CENTER, S::BOTTOM}};
            t.city = {{S::TOP}};
            t.farms = {
                FarmerConnection({S::BOTTOM_LEFT}, {FS::BLB, FS::BLL}),
                FarmerConnection({S::BOTTOM_RIGHT}, {FS::BRB, FS::BRR}),
                FarmerConnection({S::TOP_LEFT}, {FS::TLL}, {S::TOP}),
                FarmerConnection({S::TOP_RIGHT}, {FS::TRR}, {S::TOP})
            };
            T["river_city_with_road"] = std::move(t);
        }

        // river_double_city
        {
            Tile t;
            t.description = "river_double_city";
            t.river = {{S::LEFT, S::RIGHT}};
            t.city = {{S::TOP}, {S::BOTTOM}};
            t.farms = {
                FarmerConnection({S::BOTTOM_LEFT, S::BOTTOM_RIGHT},
                    {FS::BLL, FS::BRR}, {S::BOTTOM}),
                FarmerConnection({S::TOP_LEFT, S::TOP_RIGHT},
                    {FS::TLL, FS::TRR}, {S::TOP})
            };
            T["river_double_city"] = std::move(t);
        }

        // river_straight
        {
            Tile t;
            t.description = "river_straight";
            t.river = {{S::TOP, S::BOTTOM}};
            t.grass = {S::LEFT, S::RIGHT};
            t.farms = {
                FarmerConnection({S::TOP_LEFT, S::BOTTOM_LEFT},
                    {FS::TLL, FS::TLT, FS::BLL, FS::BLB}),
                FarmerConnection({S::TOP_RIGHT, S::BOTTOM_RIGHT},
                    {FS::TRR, FS::TRT, FS::BRR, FS::BRB})
            };
            T["river_straight"] = std::move(t);
        }

        // river_diagonal_city
        {
            Tile t;
            t.description = "river_diagonal_city";
            t.river = {{S::LEFT, S::BOTTOM}};
            t.city = {{S::TOP, S::RIGHT}};
            t.farms = {
                FarmerConnection({S::TOP_LEFT, S::BOTTOM_RIGHT},
                    {FS::TLL, FS::BLB}, {S::TOP, S::RIGHT}),
                FarmerConnection({S::BOTTOM_LEFT},
                    {FS::BLL, FS::BLB})
            };
            T["river_diagonal_city"] = std::move(t);
        }

        // river_straight_2
        {
            Tile t;
            t.description = "river_straight_2";
            t.river = {{S::TOP, S::BOTTOM}};
            t.grass = {S::LEFT, S::RIGHT};
            t.farms = {
                FarmerConnection({S::TOP_LEFT, S::BOTTOM_LEFT},
                    {FS::TLL, FS::TLT, FS::BLL, FS::BLB}),
                FarmerConnection({S::TOP_RIGHT, S::BOTTOM_RIGHT},
                    {FS::TRR, FS::TRT, FS::BRR, FS::BRB})
            };
            T["river_straight_2"] = std::move(t);
        }

        // river_bend
        {
            Tile t;
            t.description = "river_bend";
            t.river = {{S::TOP, S::LEFT}};
            t.grass = {S::RIGHT, S::BOTTOM};
            t.farms = {
                FarmerConnection({S::TOP_LEFT}, {FS::TLL, FS::TLT}),
                FarmerConnection({S::TOP_RIGHT, S::BOTTOM_RIGHT, S::BOTTOM_LEFT},
                    {FS::TRR, FS::TRT, FS::BRR, FS::BRB, FS::BLL, FS::BLB})
            };
            T["river_bend"] = std::move(t);
        }

        // river_chapel
        {
            Tile t;
            t.description = "river_chapel";
            t.river = {{S::LEFT, S::RIGHT}};
            t.road = {{S::CENTER, S::BOTTOM}};
            t.grass = {S::TOP};
            t.farms = {
                FarmerConnection({S::BOTTOM_LEFT}, {FS::BLL, FS::BLB}),
                FarmerConnection({S::BOTTOM_RIGHT}, {FS::BRR, FS::BRB}),
                FarmerConnection({S::TOP_LEFT, S::TOP_RIGHT},
                    {FS::TLL, FS::TLT, FS::TRR, FS::TRT})
            };
            t.chapel = true;
            T["river_chapel"] = std::move(t);
        }

        // river_bend_with_road
        {
            Tile t;
            t.description = "river_bend_with_road";
            t.river = {{S::RIGHT, S::BOTTOM}};
            t.road = {{S::LEFT, S::TOP}};
            t.farms = {
                FarmerConnection({S::TOP_LEFT}, {FS::TLL, FS::TLT}),
                FarmerConnection({S::BOTTOM_RIGHT}, {FS::BRR, FS::BRB}),
                FarmerConnection({S::BOTTOM_LEFT, S::TOP_RIGHT},
                    {FS::BLL, FS::BLB, FS::TRR, FS::TRT})
            };
            T["river_bend_with_road"] = std::move(t);
        }

        // river_flowery_bend
        {
            Tile t;
            t.description = "river_flowery_bend";
            t.river = {{S::RIGHT, S::BOTTOM}};
            t.grass = {S::LEFT, S::TOP};
            t.farms = {
                FarmerConnection({S::BOTTOM_RIGHT}, {FS::BRR, FS::BRB}),
                FarmerConnection({S::TOP_LEFT, S::BOTTOM_LEFT, S::TOP_RIGHT},
                    {FS::BLL, FS::BLB, FS::TRR, FS::TRT, FS::TLL, FS::TLT})
            };
            t.flowers = true;
            T["river_flowery_bend"] = std::move(t);
        }

        // river_crossing
        {
            Tile t;
            t.description = "river_crossing";
            t.river = {{S::TOP, S::BOTTOM}};
            t.road = {{S::LEFT, S::RIGHT}};
            t.farms = {
                FarmerConnection({S::BOTTOM_LEFT}, {FS::BLB, FS::BLL}),
                FarmerConnection({S::BOTTOM_RIGHT}, {FS::BRB, FS::BRR}),
                FarmerConnection({S::TOP_LEFT}, {FS::TLL, FS::TLT}),
                FarmerConnection({S::TOP_RIGHT}, {FS::TRR, FS::TRT})
            };
            T["river_crossing"] = std::move(t);
        }

        // river_end
        {
            Tile t;
            t.description = "river_end";
            t.river = {{S::TOP, S::CENTER}};
            t.grass = {S::LEFT, S::RIGHT, S::BOTTOM};
            t.farms = {FarmerConnection(
                {S::TOP_LEFT, S::TOP_RIGHT, S::BOTTOM_LEFT, S::BOTTOM_RIGHT},
                {FS::TLL, FS::TLT, FS::TRT, FS::TRR, FS::BRR, FS::BRB, FS::BLB, FS::BLL}
            )};
            T["river_end"] = std::move(t);
        }

        d.tile_counts = {
            {"river_start", 1}, {"river_straight", 2}, {"river_flowery_bend", 1},
            {"river_chapel", 1}, {"river_crossing", 1}, {"river_bend", 1},
            {"river_double_city", 1}, {"river_bend_with_road", 1},
            {"river_city_with_road", 1}, {"river_end", 1}
        };

        return d;
    }();
    return deck;
}

} // namespace carcassonne

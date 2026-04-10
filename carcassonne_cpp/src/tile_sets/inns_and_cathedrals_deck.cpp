#include "carcassonne/tile_sets.h"
#include "carcassonne/types.h"

namespace carcassonne {

namespace S = Side;
namespace FS = FarmerSide;

const TileDeck& get_inns_and_cathedrals_deck() {
    static TileDeck deck = []() {
        TileDeck d;
        auto& T = d.tiles;

        // ic_1: city=[LEFT,BOTTOM],[RIGHT], grass=TOP, shield
        {
            Tile t;
            t.description = "ic_1";
            t.grass = {S::TOP};
            t.city = {{S::LEFT, S::BOTTOM}, {S::RIGHT}};
            t.farms = {FarmerConnection(
                {S::TOP_LEFT, S::TOP_RIGHT},
                {FS::TLT, FS::TRT},
                {S::LEFT, S::RIGHT, S::BOTTOM}
            )};
            t.shield = true;
            T["ic_1"] = std::move(t);
        }

        // ic_2: grass=TOP,RIGHT, road=BOTTOM-LEFT
        {
            Tile t;
            t.description = "ic_2";
            t.grass = {S::TOP, S::RIGHT};
            t.road = {{S::BOTTOM, S::LEFT}};
            t.farms = {
                FarmerConnection(
                    {S::TOP_LEFT, S::TOP_RIGHT, S::BOTTOM_RIGHT},
                    {FS::TLL, FS::TLT, FS::TRR, FS::TRT, FS::BRR, FS::BRB}
                ),
                FarmerConnection({S::BOTTOM_LEFT}, {FS::BLL, FS::BLB})
            };
            T["ic_2"] = std::move(t);
        }

        // ic_3: road=BOTTOM-RIGHT, TOP-LEFT
        {
            Tile t;
            t.description = "ic_3";
            t.road = {{S::BOTTOM, S::RIGHT}, {S::TOP, S::LEFT}};
            t.farms = {
                FarmerConnection({S::TOP_LEFT}, {FS::TLL, FS::TLT}),
                FarmerConnection({S::BOTTOM_RIGHT}, {FS::BRR, FS::BRB}),
                FarmerConnection({S::BOTTOM_LEFT, S::TOP_RIGHT},
                    {FS::BLL, FS::BLB, FS::TRR, FS::TRT})
            };
            T["ic_3"] = std::move(t);
        }

        // ic_4: city=LEFT,RIGHT, road=TOP-CENTER, BOTTOM-CENTER
        {
            Tile t;
            t.description = "ic_4";
            t.city = {{S::LEFT, S::RIGHT}};
            t.road = {{S::TOP, S::CENTER}, {S::BOTTOM, S::CENTER}};
            t.farms = {
                FarmerConnection({S::TOP_LEFT}, {FS::TLT}, {S::LEFT, S::RIGHT}),
                FarmerConnection({S::TOP_RIGHT}, {FS::TRT}, {S::LEFT, S::RIGHT}),
                FarmerConnection({S::BOTTOM_LEFT}, {FS::BLB}, {S::LEFT, S::RIGHT}),
                FarmerConnection({S::BOTTOM_RIGHT}, {FS::BRB}, {S::LEFT, S::RIGHT})
            };
            T["ic_4"] = std::move(t);
        }

        // ic_5: full city with cathedral
        {
            Tile t;
            t.description = "ic_5";
            t.city = {{S::TOP, S::RIGHT, S::BOTTOM, S::LEFT}};
            t.cathedral = true;
            T["ic_5"] = std::move(t);
        }

        // ic_6: full city with cathedral
        {
            Tile t;
            t.description = "ic_6";
            t.city = {{S::TOP, S::RIGHT, S::BOTTOM, S::LEFT}};
            t.cathedral = true;
            T["ic_6"] = std::move(t);
        }

        // ic_7: city=TOP, RIGHT, LEFT, grass=BOTTOM
        {
            Tile t;
            t.description = "ic_7";
            t.city = {{S::TOP}, {S::RIGHT}, {S::LEFT}};
            t.grass = {S::BOTTOM};
            t.farms = {
                FarmerConnection({S::BOTTOM_LEFT, S::BOTTOM_RIGHT},
                    {FS::BLB, FS::BRB}, {S::LEFT, S::RIGHT, S::TOP})
            };
            T["ic_7"] = std::move(t);
        }

        // ic_8: 4 separate cities, flowers
        {
            Tile t;
            t.description = "ic_8";
            t.city = {{S::TOP}, {S::RIGHT}, {S::LEFT}, {S::BOTTOM}};
            t.farms = {
                FarmerConnection({S::CENTER}, {}, {S::LEFT, S::RIGHT, S::TOP, S::BOTTOM})
            };
            t.flowers = true;
            T["ic_8"] = std::move(t);
        }

        // ic_9: road=LEFT-RIGHT, grass=TOP,BOTTOM, flowers
        {
            Tile t;
            t.description = "ic_9";
            t.grass = {S::TOP, S::BOTTOM};
            t.road = {{S::LEFT, S::RIGHT}};
            t.farms = {
                FarmerConnection({S::TOP_LEFT, S::TOP_RIGHT},
                    {FS::TLL, FS::TLT, FS::TRR, FS::TRT}),
                FarmerConnection({S::BOTTOM_LEFT, S::BOTTOM_RIGHT},
                    {FS::BLL, FS::BLB, FS::BRR, FS::BRB})
            };
            t.flowers = true;
            T["ic_9"] = std::move(t);
        }

        // ic_10: city=LEFT, grass=TOP, road=BOTTOM-RIGHT
        {
            Tile t;
            t.description = "ic_10";
            t.city = {{S::LEFT}};
            t.grass = {S::TOP};
            t.road = {{S::BOTTOM, S::RIGHT}};
            t.farms = {
                FarmerConnection({S::TOP_LEFT, S::TOP_RIGHT, S::BOTTOM_LEFT},
                    {FS::TLT, FS::TRR, FS::TRT, FS::BLB}, {S::LEFT}),
                FarmerConnection({S::BOTTOM_RIGHT}, {FS::BRR, FS::BRB})
            };
            T["ic_10"] = std::move(t);
        }

        // ic_11: city=LEFT,TOP, grass=BOTTOM, road=CENTER-RIGHT
        {
            Tile t;
            t.description = "ic_11";
            t.city = {{S::LEFT, S::TOP}};
            t.grass = {S::BOTTOM};
            t.road = {{S::CENTER, S::RIGHT}};
            t.farms = {
                FarmerConnection({S::TOP_RIGHT}, {FS::TRR}, {S::TOP, S::LEFT}),
                FarmerConnection({S::BOTTOM_RIGHT, S::BOTTOM_LEFT},
                    {FS::BRR, FS::BRB, FS::BLB}, {S::TOP, S::LEFT})
            };
            T["ic_11"] = std::move(t);
        }

        // ic_12: city=RIGHT,TOP, grass=BOTTOM, road=CENTER-LEFT
        {
            Tile t;
            t.description = "ic_12";
            t.city = {{S::RIGHT, S::TOP}};
            t.grass = {S::BOTTOM};
            t.road = {{S::CENTER, S::LEFT}};
            t.farms = {
                FarmerConnection({S::TOP_LEFT}, {FS::TLL}, {S::TOP, S::RIGHT}),
                FarmerConnection({S::BOTTOM_RIGHT, S::BOTTOM_LEFT},
                    {FS::BRB, FS::BLL, FS::BLB}, {S::TOP, S::RIGHT})
            };
            T["ic_12"] = std::move(t);
        }

        // ic_13: city=BOTTOM, grass=LEFT,RIGHT,TOP
        {
            Tile t;
            t.description = "ic_13";
            t.city = {{S::BOTTOM}};
            t.grass = {S::LEFT, S::RIGHT, S::TOP};
            t.farms = {
                FarmerConnection({S::BOTTOM_LEFT},
                    {FS::BLL, FS::TLL}, {S::BOTTOM}),
                FarmerConnection({S::TOP_RIGHT},
                    {FS::TLT, FS::TRR, FS::TRT, FS::BRR}, {S::BOTTOM})
            };
            T["ic_13"] = std::move(t);
        }

        // ic_14: city=BOTTOM,RIGHT, road=TOP-LEFT, shield
        {
            Tile t;
            t.description = "ic_14";
            t.city = {{S::BOTTOM, S::RIGHT}};
            t.road = {{S::TOP, S::LEFT}};
            t.farms = {
                FarmerConnection({S::TOP_LEFT}, {FS::TLL, FS::TLT}),
                FarmerConnection({S::TOP_RIGHT, S::BOTTOM_LEFT},
                    {FS::TRT, FS::BLL}, {S::BOTTOM, S::RIGHT})
            };
            t.shield = true;
            T["ic_14"] = std::move(t);
        }

        // ic_15: city=BOTTOM, road=TOP-CENTER, grass=LEFT,RIGHT
        {
            Tile t;
            t.description = "ic_15";
            t.city = {{S::BOTTOM}};
            t.road = {{S::TOP, S::CENTER}};
            t.grass = {S::LEFT, S::RIGHT};
            t.farms = {
                FarmerConnection({S::TOP_LEFT, S::BOTTOM_LEFT},
                    {FS::TLL, FS::TLT, FS::BLL}, {S::BOTTOM}),
                FarmerConnection({S::TOP_RIGHT, S::BOTTOM_RIGHT},
                    {FS::TRR, FS::TRT, FS::BRR}, {S::BOTTOM})
            };
            T["ic_15"] = std::move(t);
        }

        // ic_16: city=LEFT, city=RIGHT, road=TOP-CENTER, BOTTOM-CENTER
        {
            Tile t;
            t.description = "ic_16";
            t.city = {{S::LEFT}, {S::RIGHT}};
            t.road = {{S::TOP, S::CENTER}, {S::BOTTOM, S::CENTER}};
            t.farms = {
                FarmerConnection({S::TOP_LEFT}, {FS::TLT}, {S::LEFT}),
                FarmerConnection({S::TOP_RIGHT}, {FS::TRT}, {S::RIGHT}),
                FarmerConnection({S::BOTTOM_LEFT}, {FS::BLB}, {S::LEFT}),
                FarmerConnection({S::BOTTOM_RIGHT}, {FS::BRB}, {S::RIGHT})
            };
            T["ic_16"] = std::move(t);
        }

        // ic_17: road=LEFT-CENTER, RIGHT-CENTER, grass=TOP,BOTTOM, chapel
        {
            Tile t;
            t.description = "ic_17";
            t.road = {{S::LEFT, S::CENTER}, {S::RIGHT, S::CENTER}};
            t.grass = {S::TOP, S::BOTTOM};
            t.farms = {
                FarmerConnection({S::TOP_LEFT, S::TOP_RIGHT},
                    {FS::TLL, FS::TLT, FS::TRR, FS::TRT}),
                FarmerConnection({S::BOTTOM_LEFT, S::BOTTOM_RIGHT},
                    {FS::BLL, FS::BLB, FS::BRR, FS::BRB})
            };
            t.chapel = true;
            T["ic_17"] = std::move(t);
        }

        // ic_18: 3-way split road, grass=TOP
        {
            Tile t;
            t.description = "ic_18";
            t.road = {{S::LEFT, S::CENTER}, {S::RIGHT, S::CENTER}, {S::BOTTOM, S::CENTER}};
            t.grass = {S::TOP};
            t.farms = {
                FarmerConnection({S::BOTTOM_LEFT}, {FS::BLB, FS::BLL}),
                FarmerConnection({S::BOTTOM_RIGHT}, {FS::BRB, FS::BRR}),
                FarmerConnection({S::TOP_LEFT, S::TOP_RIGHT},
                    {FS::TLL, FS::TLT, FS::TRR, FS::TRT})
            };
            T["ic_18"] = std::move(t);
        }

        d.tile_counts = {
            {"ic_1", 1}, {"ic_2", 1}, {"ic_3", 1}, {"ic_4", 1},
            {"ic_5", 1}, {"ic_6", 1}, {"ic_7", 1}, {"ic_8", 1},
            {"ic_9", 1}, {"ic_10", 1}, {"ic_11", 1}, {"ic_12", 1},
            {"ic_13", 1}, {"ic_14", 1}, {"ic_15", 1}, {"ic_16", 1},
            {"ic_17", 1}, {"ic_18", 1}
        };

        return d;
    }();
    return deck;
}

} // namespace carcassonne

#pragma once
#include "types.h"
#include "connection.h"

namespace carcassonne {

class SideModificationUtil {
public:
    static int turn_side(int side, int times) {
        if (side == Side::CENTER) return side;
        return (side + times * 2) % 8;
    }

    static int opposite_side(int side) {
        return turn_side(side, 2);
    }

    static std::vector<int> turn_sides(const std::vector<int>& sides, int times) {
        std::vector<int> result;
        result.reserve(sides.size());
        for (int s : sides) result.push_back(turn_side(s, times));
        return result;
    }

    static int turn_farmer_side(int farmer_side, int times) {
        return (farmer_side + times * 2) % 8;
    }

    static std::vector<int> turn_farmer_sides(const std::vector<int>& farmer_sides, int times) {
        std::vector<int> result;
        result.reserve(farmer_sides.size());
        for (int fs : farmer_sides) result.push_back(turn_farmer_side(fs, times));
        return result;
    }

    static int opposite_farmer_side(int farmer_side) {
        switch (farmer_side) {
            case FarmerSide::TLL: return FarmerSide::TRR;
            case FarmerSide::TLT: return FarmerSide::BLB;
            case FarmerSide::TRT: return FarmerSide::BRB;
            case FarmerSide::TRR: return FarmerSide::TLL;
            case FarmerSide::BRR: return FarmerSide::BLL;
            case FarmerSide::BRB: return FarmerSide::TRT;
            case FarmerSide::BLB: return FarmerSide::TLT;
            case FarmerSide::BLL: return FarmerSide::BRR;
        }
        return farmer_side;
    }

    static FarmerConnection turn_farmer_connection(const FarmerConnection& fc, int times) {
        return FarmerConnection(
            turn_sides(fc.farmer_positions, times),
            turn_farmer_sides(fc.tile_connections, times),
            turn_sides(fc.city_sides, times)
        );
    }

    static Connection turn_connection(const Connection& conn, int times) {
        return Connection(turn_side(conn.a, times), turn_side(conn.b, times));
    }
};

} // namespace carcassonne

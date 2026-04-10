#pragma once
#include <algorithm>
#include <array>
#include <cstring>
#include <map>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "game_state.h"
#include "tile.h"
#include "types.h"

namespace carcassonne {

constexpr int NUM_TERRAIN_TYPES = 5;
constexpr int CANON_TERRAIN_DIM = 20;
constexpr int NUM_TILE_FLAGS = 4;
constexpr int NUM_SIDES = 9;
constexpr int NUM_MEEPLE_TYPES_FEAT = 5;
constexpr int MEEPLE_OWNER_DIM = 3;
constexpr int NUM_GAME_GLOBALS = 7;
constexpr int EDGE_FEAT_DIM = 7;
constexpr int EDGE_MODEL_DIM_FEAT = 3;

constexpr int GPS_TILE_STATIC_DIM_FEAT = 50 + CANON_TERRAIN_DIM + NUM_TILE_FLAGS;

constexpr int GPS_NODE_FEAT_DIM_FEAT =
    GPS_TILE_STATIC_DIM_FEAT + 1 + 1 + MEEPLE_OWNER_DIM +
    NUM_MEEPLE_TYPES_FEAT + NUM_SIDES + NUM_GAME_GLOBALS;

inline int terrain_idx(TerrainType t) {
    switch (t) {
        case TerrainType::CITY:    return 0;
        case TerrainType::ROAD:    return 1;
        case TerrainType::GRASS:   return 2;
        case TerrainType::CHAPEL:
        case TerrainType::FLOWERS: return 3;
        default:                   return 4;
    }
}

inline int edge_terrain_idx(TerrainType t) {
    switch (t) {
        case TerrainType::CITY:  return 0;
        case TerrainType::ROAD:  return 1;
        case TerrainType::GRASS: return 2;
        default:                 return -1;
    }
}

inline int direction_idx(int side) {
    switch (side) {
        case Side::TOP:    return 0;
        case Side::RIGHT:  return 1;
        case Side::BOTTOM: return 2;
        case Side::LEFT:   return 3;
        default:           return -1;
    }
}

inline std::pair<int,int> dir_delta(int side) {
    switch (side) {
        case Side::TOP:    return {-1,  0};
        case Side::RIGHT:  return { 0,  1};
        case Side::BOTTOM: return { 1,  0};
        case Side::LEFT:   return { 0, -1};
        default:           return { 0,  0};
    }
}

static const int CARDINAL[4] = {Side::TOP, Side::RIGHT, Side::BOTTOM, Side::LEFT};

inline void terrain_one_hot(TerrainType t, float* out) {
    std::memset(out, 0, NUM_TERRAIN_TYPES * sizeof(float));
    int idx = terrain_idx(t);
    if (idx >= 0 && idx < NUM_TERRAIN_TYPES) out[idx] = 1.0f;
}

inline void canonical_terrain_sequence(const Tile& tile, float* out) {
    float sides_oh[4][NUM_TERRAIN_TYPES];
    for (int i = 0; i < 4; ++i)
        terrain_one_hot(tile.get_type(CARDINAL[i]), sides_oh[i]);

    std::array<float, CANON_TERRAIN_DIM> best;
    for (int shift = 0; shift < 4; ++shift) {
        std::array<float, CANON_TERRAIN_DIM> cand;
        for (int i = 0; i < 4; ++i)
            std::memcpy(&cand[(i) * NUM_TERRAIN_TYPES],
                        sides_oh[(shift + i) % 4],
                        NUM_TERRAIN_TYPES * sizeof(float));
        if (shift == 0 || cand < best)
            best = cand;
    }
    std::memcpy(out, best.data(), CANON_TERRAIN_DIM * sizeof(float));
}

struct GraphFeatures {
    std::vector<float> x_gps;       // (N * GPS_NODE_FEAT_DIM_FEAT)
    std::vector<int64_t> edge_index; // (2 * E)
    std::vector<float> edge_attr;   // (E * EDGE_FEAT_DIM)
    int last_placed_node_idx;
    int origin_row;
    int origin_col;
    int num_nodes;
    int num_edges;
};

inline GraphFeatures build_graph_features(
    const CarcassonneGameState& state,
    const std::unordered_map<std::string, int>& tile_desc_to_idx,
    int num_tile_types = 50
) {
    int player = state.current_player;
    int enemy = 1 - player;

    using RC = std::pair<int, int>;
    std::map<RC, const Tile*> sorted_board;
    for (auto& [pos, tile] : state.board)
        sorted_board[pos] = &tile;

    int N = static_cast<int>(sorted_board.size());
    std::unordered_map<int64_t, int> pos_to_idx;
    std::vector<RC> positions;
    positions.reserve(N);

    auto pack_key = [](int r, int c) -> int64_t {
        return (static_cast<int64_t>(r) << 32) | static_cast<uint32_t>(c);
    };

    int idx = 0;
    for (auto& [pos, tile_ptr] : sorted_board) {
        pos_to_idx[pack_key(pos.first, pos.second)] = idx++;
        positions.push_back(pos);
    }

    // Meeple map: (row, col) -> (owner, meeple_type, side)
    std::unordered_map<int64_t, std::tuple<int,int,int>> tile_meeple;
    for (int p_idx = 0; p_idx < 2 && p_idx < static_cast<int>(state.placed_meeples.size()); ++p_idx) {
        for (auto& mp : state.placed_meeples[p_idx]) {
            int r = mp.coordinate_with_side.coordinate.row;
            int c = mp.coordinate_with_side.coordinate.column;
            int s = mp.coordinate_with_side.side;
            tile_meeple[pack_key(r, c)] = {p_idx, mp.meeple_type, s};
        }
    }

    RC last_placed_pos = {-99999, -99999};
    if (state.last_tile_action.has_value()) {
        last_placed_pos = {
            state.last_tile_action->coordinate.row,
            state.last_tile_action->coordinate.column
        };
    }

    int last_placed_node_idx = -1;
    {
        auto it = pos_to_idx.find(pack_key(last_placed_pos.first, last_placed_pos.second));
        if (it != pos_to_idx.end()) last_placed_node_idx = it->second;
    }

    float pm = (player < static_cast<int>(state.meeples.size())) ? state.meeples[player] : 0;
    float pa = (player < static_cast<int>(state.abbots.size())) ? state.abbots[player] : 0;
    float pb = (player < static_cast<int>(state.big_meeples.size())) ? state.big_meeples[player] : 0;
    float em = (enemy < static_cast<int>(state.meeples.size())) ? state.meeples[enemy] : 0;
    float ea = (enemy < static_cast<int>(state.abbots.size())) ? state.abbots[enemy] : 0;
    float eb = (enemy < static_cast<int>(state.big_meeples.size())) ? state.big_meeples[enemy] : 0;
    float is_meeple_phase = (state.phase == GamePhase::MEEPLES) ? 1.0f : 0.0f;

    GraphFeatures gf;
    gf.num_nodes = N;
    gf.last_placed_node_idx = last_placed_node_idx;
    if (N > 0) {
        gf.origin_row = positions[0].first;
        gf.origin_col = positions[0].second;
    } else {
        gf.origin_row = 0;
        gf.origin_col = 0;
    }

    gf.x_gps.resize(N * GPS_NODE_FEAT_DIM_FEAT, 0.0f);

    for (int i = 0; i < N; ++i) {
        auto& [r, c] = positions[i];
        auto it = sorted_board.find({r, c});
        const Tile& tile = *(it->second);
        float* buf = &gf.x_gps[i * GPS_NODE_FEAT_DIM_FEAT];
        int off = 0;

        // tile-type one-hot
        auto desc_it = tile_desc_to_idx.find(tile.description);
        if (desc_it != tile_desc_to_idx.end() && desc_it->second >= 0 && desc_it->second < num_tile_types)
            buf[off + desc_it->second] = 1.0f;
        off += num_tile_types;

        // canonical terrain sequence
        canonical_terrain_sequence(tile, buf + off);
        off += CANON_TERRAIN_DIM;

        // flags
        buf[off + 0] = tile.chapel ? 1.0f : 0.0f;
        buf[off + 1] = tile.shield ? 1.0f : 0.0f;
        buf[off + 2] = tile.cathedral ? 1.0f : 0.0f;
        buf[off + 3] = tile.flowers ? 1.0f : 0.0f;
        off += NUM_TILE_FLAGS;

        // just_placed
        buf[off] = (r == last_placed_pos.first && c == last_placed_pos.second) ? 1.0f : 0.0f;
        off += 1;

        // meeple info
        int mo = -1, mt = -1, ms = -1;
        auto m_it = tile_meeple.find(pack_key(r, c));
        if (m_it != tile_meeple.end()) {
            mo = std::get<0>(m_it->second);
            mt = std::get<1>(m_it->second);
            ms = std::get<2>(m_it->second);
        }

        buf[off] = (mo >= 0) ? 1.0f : 0.0f;
        off += 1;

        // meeple_owner one-hot
        if (mo == 0)      buf[off + 1] = 1.0f;
        else if (mo == 1) buf[off + 2] = 1.0f;
        else              buf[off + 0] = 1.0f;
        off += MEEPLE_OWNER_DIM;

        // meeple_type one-hot
        if (mt >= 0 && mt < NUM_MEEPLE_TYPES_FEAT)
            buf[off + mt] = 1.0f;
        off += NUM_MEEPLE_TYPES_FEAT;

        // meeple_side one-hot
        if (ms >= 0 && ms < NUM_SIDES)
            buf[off + ms] = 1.0f;
        off += NUM_SIDES;

        // game globals
        buf[off + 0] = pm / 7.0f;
        buf[off + 1] = pa;
        buf[off + 2] = pb;
        buf[off + 3] = em / 7.0f;
        buf[off + 4] = ea;
        buf[off + 5] = eb;
        buf[off + 6] = is_meeple_phase;
    }

    // Build edges
    std::vector<int64_t> edge_src, edge_dst;
    std::vector<float> edge_feats_flat;

    for (int i = 0; i < N; ++i) {
        auto& [r, c] = positions[i];
        auto board_it = sorted_board.find({r, c});
        const Tile& tile = *(board_it->second);

        for (int d = 0; d < 4; ++d) {
            int direction = CARDINAL[d];
            auto [dr, dc] = dir_delta(direction);
            auto j_it = pos_to_idx.find(pack_key(r + dr, c + dc));
            if (j_it == pos_to_idx.end()) continue;

            edge_src.push_back(i);
            edge_dst.push_back(j_it->second);

            float ef[EDGE_FEAT_DIM] = {};
            TerrainType terrain = tile.get_type(direction);
            int eidx = edge_terrain_idx(terrain);
            if (eidx >= 0) ef[eidx] = 1.0f;
            int didx = direction_idx(direction);
            if (didx >= 0) ef[3 + didx] = 1.0f;

            for (int k = 0; k < EDGE_FEAT_DIM; ++k)
                edge_feats_flat.push_back(ef[k]);
        }
    }

    int E = static_cast<int>(edge_src.size());
    gf.num_edges = E;
    gf.edge_index.resize(2 * E);
    for (int i = 0; i < E; ++i) {
        gf.edge_index[i] = edge_src[i];
        gf.edge_index[E + i] = edge_dst[i];
    }
    gf.edge_attr = std::move(edge_feats_flat);

    return gf;
}

} // namespace carcassonne

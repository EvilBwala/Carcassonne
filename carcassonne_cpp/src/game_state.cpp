#include "carcassonne/game_state.h"

namespace carcassonne {

TileRegistry& get_global_tile_registry() {
    static TileRegistry registry = []() {
        TileRegistry r;
        r.add_deck(get_base_deck().tiles);
        r.add_deck(get_inns_and_cathedrals_deck().tiles);
        r.add_deck(get_the_river_deck().tiles);
        return r;
    }();
    return registry;
}

} // namespace carcassonne

cmake_minimum_required(VERSION 3.15)
project(carcassonne_engine CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(FetchContent)

FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.11.1
)
FetchContent_MakeAvailable(pybind11)

pybind11_add_module(_carcassonne_engine
    src/bindings.cpp
    src/game_state.cpp
    src/tile_sets/base_deck.cpp
    src/tile_sets/inns_and_cathedrals_deck.cpp
    src/tile_sets/the_river_deck.cpp
)

target_include_directories(_carcassonne_engine PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_options(_carcassonne_engine PRIVATE
    -O2 -Wall -Wextra -Wno-unused-parameter
)

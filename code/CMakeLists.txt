cmake_minimum_required(VERSION 3.5.1)
project(celerity)

find_package(celerity CONFIG REQUIRED)
add_executable(abbc src/abbc.cpp)
add_celerity_to_target(TARGET abbc SOURCES src/abbc.cpp)

add_executable(abbc_commutative src/commutative/abbc.cpp)
add_celerity_to_target(TARGET abbc_commutative SOURCES src/commutative/abbc.cpp)

add_executable(mvt src/mvt.cpp)
add_celerity_to_target(TARGET mvt SOURCES src/mvt.cpp)

add_executable(mvt_commutative src/commutative/mvt.cpp)
add_celerity_to_target(TARGET mvt_commutative SOURCES src/commutative/mvt.cpp)


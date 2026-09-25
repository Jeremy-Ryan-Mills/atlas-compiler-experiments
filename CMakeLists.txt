cmake_minimum_required(VERSION 3.20)
project(atlas_opt CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
add_compile_options(-Wall -Wextra -Wpedantic)
option(ATLAS_SANITIZE "Build with AddressSanitizer and UBSan" OFF)
if(ATLAS_SANITIZE)
  add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
  add_link_options(-fsanitize=address,undefined)
endif()

# New files in src/core or src/passes are picked up automatically.
file(GLOB CORE_SOURCES CONFIGURE_DEPENDS src/core/*.cpp)
file(GLOB PASS_SOURCES CONFIGURE_DEPENDS src/passes/*.cpp)
add_library(atlas STATIC ${CORE_SOURCES} ${PASS_SOURCES} src/tool/viewer.cpp)
target_include_directories(atlas PUBLIC src)

add_executable(atlas-opt src/tool/main.cpp)
target_link_libraries(atlas-opt atlas)

enable_testing()
add_executable(atlas-tests tests/tests.cpp)
target_link_libraries(atlas-tests atlas)
target_compile_definitions(atlas-tests PRIVATE ATLAS_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
add_test(NAME atlas-tests COMMAND atlas-tests)

add_executable(publication-fixed-tests tests/publication_fixed_tests.cpp)
target_link_libraries(publication-fixed-tests atlas)
add_test(NAME publication-fixed-tests COMMAND publication-fixed-tests)

add_executable(publication-dma-tests tests/publication_dma_tests.cpp)
target_link_libraries(publication-dma-tests atlas)
add_test(NAME publication-dma-tests COMMAND publication-dma-tests)

add_executable(dma-wait-insertion-tests tests/dma_wait_insertion_tests.cpp)
target_link_libraries(dma-wait-insertion-tests atlas)
add_test(NAME dma-wait-insertion-tests COMMAND dma-wait-insertion-tests)

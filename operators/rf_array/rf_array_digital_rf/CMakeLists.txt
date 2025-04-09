# SPDX-FileCopyrightText: 2025 Massachusetts Institute of Technology
#
# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.20)
project(rf_array_digital_rf CXX)

find_package(holoscan 2.5.0 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

# Enable cuda language
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()
enable_language(CUDA)

# Download external dependencies
include(FetchContent)
FetchContent_Declare(libdigital_rf
  GIT_REPOSITORY https://github.com/MITHaystack/digital_rf
  GIT_TAG 2.6.11
  SOURCE_SUBDIR c
)
FetchContent_MakeAvailable(libdigital_rf)

if (NOT TARGET holoscan::ops::rf_array)
  add_subdirectory(../rf_array ../rf_array)
endif()

add_library(rf_array_digital_rf_sink SHARED
  digital_rf_sink.cu
)
add_library(holoscan::ops::rf_array_digital_rf_sink ALIAS rf_array_digital_rf_sink)
target_include_directories(rf_array_digital_rf_sink
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_link_libraries(rf_array_digital_rf_sink
  PUBLIC
    digital_rf::digital_rf
    holoscan::core
    holoscan::ops::rf_array
    matx::matx
)

if(HOLOHUB_BUILD_PYTHON)
    add_subdirectory(python)
endif()

install(
    TARGETS
      rf_array_digital_rf_sink
)

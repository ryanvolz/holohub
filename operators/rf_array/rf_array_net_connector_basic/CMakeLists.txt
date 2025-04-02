# SPDX-FileCopyrightText: 2025 Massachusetts Institute of Technology
#
# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.20)
project(rf_array_net_connector_basic CXX)

find_package(holoscan 2.5.0 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

# Enable cuda language
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()
enable_language(CUDA)

if (NOT TARGET holoscan::ops::rf_array_header)
  add_subdirectory(../rf_array_header ../rf_array_header)
endif()

if (NOT TARGET rf_array_net_connector_common)
  add_subdirectory(../rf_array_net_connector_common ../rf_array_net_connector_common)
endif()

add_library(rf_array_net_connector_basic SHARED
  net_connector_basic.cu
)
add_library(holoscan::ops::rf_array_net_connector_basic ALIAS rf_array_net_connector_basic)
target_include_directories(rf_array_net_connector_basic
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(rf_array_net_connector_basic
  PUBLIC
    holoscan::core
    holoscan::ops::basic_network
    holoscan::ops::rf_array_header
    matx::matx
    rf_array_net_connector_common
)

install(
    TARGETS
      rf_array_net_connector_basic
)

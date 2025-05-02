# SPDX-FileCopyrightText: 2025 Massachusetts Institute of Technology
#
# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.20)
project(rf_array_subchannel_select CXX)

find_package(holoscan 2.5.0 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

# Enable cuda language
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()
enable_language(CUDA)

if (NOT TARGET holoscan::ops::rf_array)
  add_subdirectory(../rf_array ../rf_array)
endif()

add_library(rf_array_subchannel_select SHARED
  subchannel_select.cu
)
add_library(holoscan::ops::rf_array_subchannel_select ALIAS rf_array_subchannel_select)
target_include_directories(rf_array_subchannel_select
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_link_libraries(rf_array_subchannel_select
  PUBLIC
    holoscan::core
    holoscan::ops::rf_array
    matx::matx
)

if(HOLOHUB_BUILD_PYTHON)
    add_subdirectory(python)
endif()

install(
    TARGETS
      rf_array_subchannel_select
    COMPONENT rf_array-cpp
)

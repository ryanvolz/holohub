# SPDX-FileCopyrightText: 2025 Massachusetts Institute of Technology
#
# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.20)
project(rf_array_header CXX)

# Enable cuda language
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()
enable_language(CUDA)

# Download external dependencies
include(FetchContent)
FetchContent_Declare(
  MatX
  GIT_REPOSITORY https://github.com/ryanvolz/MatX.git
  GIT_TAG mimo_radar
)
FetchContent_MakeAvailable(MatX)

add_library(rf_array_header INTERFACE include/rf_array/rf_array.h)
add_library(holoscan::ops::rf_array_header ALIAS rf_array_header)
target_include_directories(rf_array_header INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(rf_array_header
  INTERFACE
    matx::matx
)

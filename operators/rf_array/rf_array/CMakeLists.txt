# SPDX-FileCopyrightText: 2025 Massachusetts Institute of Technology
#
# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.20)
project(rf_array CXX)

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

add_library(rf_array INTERFACE include/rf_array/rf_array.h)
add_library(holoscan::ops::rf_array ALIAS rf_array)
target_include_directories(rf_array INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(rf_array
  INTERFACE
    matx::matx
)

if(HOLOHUB_BUILD_PYTHON)
    add_subdirectory(python)
endif()

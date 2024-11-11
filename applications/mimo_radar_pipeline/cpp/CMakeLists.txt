# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
cmake_minimum_required(VERSION 3.20)
project(mimo_radar_pipeline CXX)

find_package(holoscan 0.6 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

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
FetchContent_Declare(libdigital_rf
  GIT_REPOSITORY https://github.com/MITHaystack/digital_rf
  GIT_TAG 2.6.8
  SOURCE_SUBDIR c
)
FetchContent_MakeAvailable(MatX libdigital_rf)

# Main
add_executable(mimo_radar_pipeline
  main.cpp
  process.cu
)

add_subdirectory(advanced_network_connectors)

target_link_libraries(mimo_radar_pipeline
  digital_rf::digital_rf
  matx::matx
  holoscan::core
  holoscan::advanced_network_rx
  holoscan::ops::ping_rx
  advanced_network_connector_rx
)

target_include_directories(mimo_radar_pipeline PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)

# Copy config file
add_custom_target(mep_yaml
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/../mep.yaml" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/../mep.yaml"
)
add_dependencies(mimo_radar_pipeline mep_yaml)

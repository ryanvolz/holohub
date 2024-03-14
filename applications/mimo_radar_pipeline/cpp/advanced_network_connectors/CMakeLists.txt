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

find_package(holoscan 0.6 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

# Enable cuda language
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()
enable_language(CUDA)

# Main
add_library(advanced_network_connector_rx STATIC
  adv_networking_rx.cu
)

set_target_properties(advanced_network_connector_rx PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_link_libraries(advanced_network_connector_rx PUBLIC
  matx::matx
  holoscan::advanced_network_rx
)

# target_include_directories(advanced_network_connector_rx PRIVATE
#   ${CMAKE_CURRENT_SOURCE_DIR}/..
#   ${CMAKE_CURRENT_SOURCE_DIR}/../../../../operators/advanced_network
# )

if(HOLOHUB_BUILD_PYTHON)
    add_subdirectory(python)
endif()

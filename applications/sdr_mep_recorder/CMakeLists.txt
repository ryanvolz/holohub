# SPDX-FileCopyrightText: Copyright (c) 2025 Massachusetts Institute of Technology
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
project(sdr_mep_recorder CXX)

find_package(holoscan 2.5.0 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

# Enable cuda language
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()
enable_language(CUDA)

# Main
add_executable(sdr_mep_recorder
  main.cpp
)

target_link_libraries(sdr_mep_recorder
  holoscan::core
  holoscan::ops::basic_network
  holoscan::ops::rf_array_digital_rf_sink
  holoscan::ops::rf_array_net_connector_basic
  holoscan::ops::rf_array_resample_poly
  holoscan::ops::rf_array_rotator_scheduled
  holoscan::ops::rf_array_subchannel_select
  holoscan::ops::rf_array_type_conversion_sc16_fc32
)

target_include_directories(sdr_mep_recorder PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)

file(GLOB SDR_MEP_RECORDER_CONFIG_FILES CONFIGURE_DEPENDS configs/*.yaml)
add_custom_target(sdr_mep_recorder_configs
  COMMAND
    ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/configs/ $<TARGET_FILE_DIR:sdr_mep_recorder>
  DEPENDS
    ${SDR_MEP_RECORDER_CONFIG_FILES}
)
add_dependencies(sdr_mep_recorder sdr_mep_recorder_configs)

install(PROGRAMS sdr_mep_recorder.py
        TYPE BIN
        COMPONENT sdr_mep_recorder
)

install(DIRECTORY configs/
        DESTINATION share/holoscan/sdr_mep_recorder/configs/
        COMPONENT sdr_mep_recorder
        FILES_MATCHING PATTERN "*.yaml"
)

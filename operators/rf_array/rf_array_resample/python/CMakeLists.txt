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

# Add on to the rf_array_python target already created in the rf_array/python directory
# to build a unified module. To do that, we need to:
# 1) add a #define that will include the binding call that has been added to rf_array_pybind.cpp,
# 2) add the sources that define the binding function, and
# 3) link to the operator definition
target_compile_definitions(rf_array_python
    PUBLIC RF_ARRAY_RESAMPLE
)

target_sources(rf_array_python
    PUBLIC resample.cpp
)

target_link_libraries(rf_array_python
    PUBLIC rf_array_resample_poly
)

add_custom_target(rf_array_resample_python_params ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/params ${HOLOHUB_PYTHON_MODULE_OUT_DIR}/rf_array/params
)

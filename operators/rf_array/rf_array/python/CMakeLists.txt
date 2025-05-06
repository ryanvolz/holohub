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

# use holohub module to ensure pybind11 is available in CMake,
# but we don't actually use the helper function
include(pybind11_add_holohub_module)

pybind11_add_holohub_module(
    CPP_CMAKE_TARGET rf_array
    CLASS_NAME "*"
    SOURCES rf_array_pybind.cpp
)

add_custom_target(rf_array_python_params
    COMMAND ${CMAKE_COMMAND} -E copy_directory params ${HOLOHUB_PYTHON_MODULE_OUT_DIR}/rf_array/params
)

install(
    DIRECTORY "${HOLOHUB_PYTHON_MODULE_OUT_DIR}/rf_array"
    DESTINATION python/lib
    FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
    DIRECTORY_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
    COMPONENT rf_array-py
    PATTERN "__pycache__" EXCLUDE
)

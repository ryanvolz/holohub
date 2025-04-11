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
include(pybind11_add_holohub_module)

pybind11_add_holohub_module(
    CPP_CMAKE_TARGET basic_network
    CLASS_NAME "NetworkOpBurstParams, BasicNetworkOpRx, BasicNetworkOpTx"
    SOURCES basic_network_pybind.cpp
)

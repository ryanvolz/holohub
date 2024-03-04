# SPDX-FileCopyrightText: Copyright (c) 2024 Massachusetts Institute of Technology. All rights reserved.
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

include(pybind11_add_holohub_module)

# Define and create a directory to store python operators under the holohub python module
if(NOT CMAKE_INSTALL_LIBDIR)
  set(CMAKE_INSTALL_LIBDIR lib)
endif()
set(HOLOHUB_PYTHON_MODULE_OUT_DIR ${CMAKE_BINARY_DIR}/python/${CMAKE_INSTALL_LIBDIR}/holohub)
file(MAKE_DIRECTORY ${HOLOHUB_PYTHON_MODULE_OUT_DIR})

pybind11_add_holohub_module(
    CPP_CMAKE_TARGET advanced_network_connector_rx
    CLASS_NAME "AdvConnectorOpRx"
    SOURCES adv_networking_rx_pybind.cpp
)

set_source_files_properties(adv_networking_rx_pybind.cpp PROPERTIES LANGUAGE CUDA)


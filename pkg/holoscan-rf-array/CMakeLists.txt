# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

set(CPACK_DEBIAN_PACKAGE_SHLIBDEPS ON)

holohub_configure_deb(
  NAME "holoscan-rf-array"
  DESCRIPTION "Holoscan RF Array operators"
  VERSION "0.3.0"
  VENDOR "MIT Haystack Observatory"
  CONTACT "Ryan Volz <rvolz@mit.edu>"
  COMPONENTS
    advanced_network-cpp
    advanced_network-py
    basic_network-cpp
    basic_network-py
    C  # libdigital_rf
    rf_array-cpp
    rf_array-py
  DEPENDS "holoscan (>= 2.9), holoscan (<< 4.0), python3-numpy, python3-scipy"
)

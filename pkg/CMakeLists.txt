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

include(holohub_configure_deb)

add_holohub_package(holoscan-networking
                    APPLICATIONS adv_networking_bench basic_networking_ping
                    OPERATORS advanced_network basic_network)
add_holohub_package(holoscan-rf-array
                    OPERATORS advanced_network
                              basic_network
                              rf_array_digital_metadata
                              rf_array_digital_rf
                              rf_array_net_connector_advanced
                              rf_array_net_connector_basic
                              rf_array_resample
                              rf_array_rotator
                              rf_array_subchannel_select
                              rf_array_type_conversion)

/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <holoscan/operators/ping_rx/ping_rx.hpp>

#include "advanced_network_connectors/adv_networking_rx.h"
#include "common.h"
#include "holoscan/holoscan.hpp"
#include "process.h"

class App : public holoscan::Application {
 private:
  /**
   * @brief Setup the application as a radar signal processing pipeline
   */
  void setup_rx() {
    using namespace holoscan;
    HOLOSCAN_LOG_INFO("Initializing radar pipeline as data processor");

    auto rx = make_operator<ops::PingRxOp>("rx");

    // Radar algorithms
    auto converter0 =
        make_operator<ops::ComplexIntToFloatOp>("converter0", from_config("radar_pipeline"));
    auto converter1 =
        make_operator<ops::ComplexIntToFloatOp>("converter1", from_config("radar_pipeline"));

    auto drf_sink0 =
        make_operator<ops::DigitalRFSinkOp>("drf_sink0", from_config("digital_rf_ch0"));
    auto drf_sink1 =
        make_operator<ops::DigitalRFSinkOp>("drf_sink1", from_config("digital_rf_ch1"));

    // Network operators
    // Advanced
    auto adv_net_rx =
        make_operator<ops::AdvNetworkOpRx>("adv_network_rx",
                                           from_config("advanced_network"),
                                           make_condition<BooleanCondition>("is_alive", true));
    auto adv_rx_pkt0 = make_operator<ops::AdvConnectorOpRx>(
        "adv_connector_rx0", from_config("rx_params"), from_config("radar_pipeline"));
    auto adv_rx_pkt1 = make_operator<ops::AdvConnectorOpRx>(
        "adv_connector_rx1", from_config("rx_params"), from_config("radar_pipeline"));

    add_flow(adv_net_rx, adv_rx_pkt0, {{"ch0", "burst_in"}});
    add_flow(adv_rx_pkt0, converter0, {{"rf_out", "rf_in"}});
    add_flow(converter0, rx);

    add_flow(adv_net_rx, adv_rx_pkt1, {{"ch1", "burst_in"}});
    add_flow(adv_rx_pkt1, converter1, {{"rf_out", "rf_in"}});
    add_flow(converter1, rx);

    add_flow(adv_rx_pkt0, drf_sink0);
    add_flow(adv_rx_pkt1, drf_sink1);
  }

 public:
  void compose() {
    using namespace holoscan;

    setup_rx();
  }
};

int main(int argc, char** argv) {
  auto app = holoscan::make_application<App>();

  // Get the configuration
  if (argc < 2) {
    HOLOSCAN_LOG_ERROR("Usage: {} [mimo_radar_pipeline.yaml]", argv[0]);
    return -1;
  }

  auto config_path = std::filesystem::canonical(argv[0]).parent_path();
  config_path += "/" + std::string(argv[1]);
  app->config(config_path);
  app->scheduler(app->make_scheduler<holoscan::MultiThreadScheduler>(
        "multithread-scheduler", app->from_config("scheduler")));
  app->run();

  return 0;
}

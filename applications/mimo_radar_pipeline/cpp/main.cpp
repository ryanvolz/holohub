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

    auto adv_net_rx =
        make_operator<ops::AdvNetworkOpRx>("adv_network_rx",
                                           from_config("advanced_network"),
                                           make_condition<BooleanCondition>("is_alive", true));
    auto adv_rx_pkt0 =
        make_operator<ops::AdvConnectorOpRx>("adv_connector_rx0", from_config("rx_params"));
    auto adv_rx_pkt1 =
        make_operator<ops::AdvConnectorOpRx>("adv_connector_rx1", from_config("rx_params"));

    auto subchannel_select0 = make_operator<ops::SubchannelSelectOp<sample_t>>(
        "subchannel_select0", from_config("SubchannelSelectOp"));
    auto subchannel_select1 = make_operator<ops::SubchannelSelectOp<sample_t>>(
        "subchannel_select1", from_config("SubchannelSelectOp"));

    // auto converter0 = make_operator<ops::ComplexIntToFloatOp>("converter0");
    auto converter1 = make_operator<ops::ComplexIntToFloatOp>("converter1");

    // auto resample0 = make_operator<ops::ResamplePolyOp>("resample0",
    // from_config("ResamplePolyOp"));
    auto resample1 = make_operator<ops::ResamplePolyOp>("resample1", from_config("ResamplePolyOp"));

    auto drf_sink0 = make_operator<ops::DigitalRFSinkOp<sample_t>>(
        "drf_sink0", from_config("DigitalRFSinkOp_emvsis"));
    auto drf_sink1 = make_operator<ops::DigitalRFSinkOp<complex_t>>(
        "drf_sink1", from_config("DigitalRFSinkOp_zephyr"));

    add_flow(adv_net_rx, adv_rx_pkt0, {{"ch1", "burst_in"}});
    add_flow(adv_rx_pkt0, subchannel_select0);
    // add_flow(subchannel_select0, converter0);
    // add_flow(converter0, resample0);
    // add_flow(resample0, drf_sink0);
    add_flow(subchannel_select0, drf_sink0);

    add_flow(adv_net_rx, adv_rx_pkt1, {{"ch0", "burst_in"}});
    add_flow(adv_rx_pkt1, subchannel_select1);
    add_flow(subchannel_select1, converter1);
    add_flow(converter1, resample1);
    add_flow(resample1, drf_sink1);
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

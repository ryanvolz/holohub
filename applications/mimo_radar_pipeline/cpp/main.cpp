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

    std::shared_ptr<holoscan::Operator> last_op;
    std::shared_ptr<holoscan::Operator> branch_op;

    auto adv_net_rx =
        make_operator<ops::AdvNetworkOpRx>("adv_network_rx",
                                           from_config("advanced_network"),
                                           make_condition<BooleanCondition>("is_alive", true));

    // sample flow 0
    auto adv_rx_pkt0 =
        make_operator<ops::AdvConnectorOpRx>("adv_connector_rx0", from_config("rx_params"));
    add_flow(adv_net_rx, adv_rx_pkt0, {{"ch0", "burst_in"}});
    last_op = adv_rx_pkt0;

    if (from_config("pipeline.subchannel_select0").as<bool>()) {
      auto subchannel_select0 = make_operator<ops::SubchannelSelectOp<sample_t>>(
          "subchannel_select0", from_config("SubchannelSelectOp"));
      add_flow(last_op, subchannel_select0);
      last_op = subchannel_select0;
    }
    if (from_config("pipeline.converter0").as<bool>()) {
      auto converter0 = make_operator<ops::ComplexIntToFloatOp>("converter0");
      add_flow(last_op, converter0);
      last_op = converter0;
      branch_op = last_op;

      if (from_config("pipeline.rotator0").as<bool>()) {
        auto rotator0 = make_operator<ops::ScheduledRotatorOp>(
            "rotator0", from_config("ScheduledRotatorOp0"));
        add_flow(branch_op, rotator0);
        last_op = rotator0;
      }

      if (from_config("pipeline.resample0").as<bool>()) {
        auto resample0 =
            make_operator<ops::ResamplePolyOp>("resample0", from_config("ResamplePolyOp0"));
        add_flow(last_op, resample0);
        last_op = resample0;
      }

      auto drf_sink0 = make_operator<ops::DigitalRFSinkOp<complex_t>>(
          "drf_sink0", from_config("DigitalRFSinkOp0"));
      add_flow(last_op, drf_sink0);

    } else {
      auto drf_sink0 = make_operator<ops::DigitalRFSinkOp<sample_t>>(
          "drf_sink0", from_config("DigitalRFSinkOp0"));
      add_flow(last_op, drf_sink0);
    }

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
    HOLOSCAN_LOG_ERROR("Usage: {} [mep.yaml]", argv[0]);
    return -1;
  }

  auto config_path = std::filesystem::current_path();
  config_path += "/" + std::string(argv[1]);
  if (!std::filesystem::exists(config_path)) {
    config_path = std::filesystem::canonical(argv[0]).parent_path();
    config_path += "/" + std::string(argv[1]);
  }
  app->config(config_path);
  app->scheduler(app->make_scheduler<holoscan::MultiThreadScheduler>(
        "multithread-scheduler", app->from_config("scheduler")));
  app->run();

  return 0;
}

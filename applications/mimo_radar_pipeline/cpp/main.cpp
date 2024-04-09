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

      // A
      if (from_config("pipeline.rotator0a").as<bool>()) {
        auto rotator0a = make_operator<ops::ScheduledRotatorOp>(
            "rotator0a", from_config("ScheduledRotatorOp0a"));
        add_flow(branch_op, rotator0a);
        last_op = rotator0a;
      }

      if (from_config("pipeline.resample0a").as<bool>()) {
        auto resample0a =
            make_operator<ops::ResamplePolyOp>("resample0a", from_config("ResamplePolyOp0"));
        add_flow(last_op, resample0a);
        last_op = resample0a;
      }

      auto drf_sink0a = make_operator<ops::DigitalRFSinkOp<complex_t>>(
          "drf_sink0a", from_config("DigitalRFSinkOp0a"));
      add_flow(last_op, drf_sink0a);

      // B
      if (from_config("pipeline.rotator0b").as<bool>()) {
        auto rotator0b = make_operator<ops::ScheduledRotatorOp>(
            "rotator0b", from_config("ScheduledRotatorOp0b"));
        add_flow(branch_op, rotator0b);
        last_op = rotator0b;
      }

      if (from_config("pipeline.resample0b").as<bool>()) {
        auto resample0b =
            make_operator<ops::ResamplePolyOp>("resample0b", from_config("ResamplePolyOp0"));
        add_flow(last_op, resample0b);
        last_op = resample0b;
      }

      auto drf_sink0b = make_operator<ops::DigitalRFSinkOp<complex_t>>(
          "drf_sink0b", from_config("DigitalRFSinkOp0b"));
      add_flow(last_op, drf_sink0b);

      // C
      if (from_config("pipeline.rotator0c").as<bool>()) {
        auto rotator0c = make_operator<ops::ScheduledRotatorOp>(
            "rotator0c", from_config("ScheduledRotatorOp0c"));
        add_flow(branch_op, rotator0c);
        last_op = rotator0c;
      }

      if (from_config("pipeline.resample0c").as<bool>()) {
        auto resample0c =
            make_operator<ops::ResamplePolyOp>("resample0c", from_config("ResamplePolyOp0"));
        add_flow(last_op, resample0c);
        last_op = resample0c;
      }

      auto drf_sink0c = make_operator<ops::DigitalRFSinkOp<complex_t>>(
          "drf_sink0c", from_config("DigitalRFSinkOp0c"));
      add_flow(last_op, drf_sink0c);

      // D
      if (from_config("pipeline.rotator0d").as<bool>()) {
        auto rotator0d = make_operator<ops::ScheduledRotatorOp>(
            "rotator0d", from_config("ScheduledRotatorOp0d"));
        add_flow(branch_op, rotator0d);
        last_op = rotator0d;
      }

      if (from_config("pipeline.resample0d").as<bool>()) {
        auto resample0d =
            make_operator<ops::ResamplePolyOp>("resample0d", from_config("ResamplePolyOp0"));
        add_flow(last_op, resample0d);
        last_op = resample0d;
      }

      auto drf_sink0d = make_operator<ops::DigitalRFSinkOp<complex_t>>(
          "drf_sink0d", from_config("DigitalRFSinkOp0d"));
      add_flow(last_op, drf_sink0d);
    } else {
      auto drf_sink0 = make_operator<ops::DigitalRFSinkOp<sample_t>>(
          "drf_sink0", from_config("DigitalRFSinkOp0"));
      add_flow(last_op, drf_sink0);
    }

    // sample flow 1
    auto adv_rx_pkt1 =
        make_operator<ops::AdvConnectorOpRx>("adv_connector_rx1", from_config("rx_params"));
    add_flow(adv_net_rx, adv_rx_pkt1, {{"ch1", "burst_in"}});
    last_op = adv_rx_pkt1;

    if (from_config("pipeline.subchannel_select1").as<bool>()) {
      auto subchannel_select1 = make_operator<ops::SubchannelSelectOp<sample_t>>(
          "subchannel_select1", from_config("SubchannelSelectOp"));
      add_flow(last_op, subchannel_select1);
      last_op = subchannel_select1;
    }
    if (from_config("pipeline.converter1").as<bool>()) {
      auto converter1 = make_operator<ops::ComplexIntToFloatOp>("converter1");
      add_flow(last_op, converter1);
      last_op = converter1;

      if (from_config("pipeline.rotator1").as<bool>()) {
        auto rotator1 =
            make_operator<ops::ScheduledRotatorOp>("rotator1", from_config("ScheduledRotatorOp1"));
        add_flow(last_op, rotator1);
        last_op = rotator1;
      }

      if (from_config("pipeline.resample1").as<bool>()) {
        auto resample1 =
            make_operator<ops::ResamplePolyOp>("resample1", from_config("ResamplePolyOp1"));
        add_flow(last_op, resample1);
        last_op = resample1;
      }

      auto drf_sink1 = make_operator<ops::DigitalRFSinkOp<complex_t>>(
          "drf_sink1", from_config("DigitalRFSinkOp1"));
      add_flow(last_op, drf_sink1);
    } else {
      auto drf_sink1 = make_operator<ops::DigitalRFSinkOp<sample_t>>(
          "drf_sink1", from_config("DigitalRFSinkOp1"));
      add_flow(last_op, drf_sink1);
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

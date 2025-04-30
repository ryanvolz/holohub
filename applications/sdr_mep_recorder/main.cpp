/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 Massachusetts Institute of Technology
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

#include "basic_network_operator_rx.h"
#include "holoscan/holoscan.hpp"
#include "rf_array/digital_rf_sink.h"
#include "rf_array/net_connector_basic.h"
#include "rf_array/resample_poly.h"
#include "rf_array/rotator_scheduled.h"
#include "rf_array/subchannel_select.h"
#include "rf_array/type_conversion_sc16_fc32.h"

class App : public holoscan::Application {
 private:
  /**
   * @brief Setup the application as a Mobile Experiment Platform (MEP) SDR recording pipeline
   */
  void setup_rx() {
    using namespace holoscan;
    HOLOSCAN_LOG_INFO("Initializing MEP recording pipeline");

    std::shared_ptr<holoscan::Operator> last_op;
    std::shared_ptr<holoscan::Operator> branch_op;

    // auto adv_net_rx =
    //     make_operator<ops::AdvNetworkOpRx>("adv_network_rx",
    //                                        from_config("advanced_network"),
    //                                        make_condition<BooleanCondition>("is_alive", true));

    // // sample flow 0
    // auto net_connector_rx0 =
    //     make_operator<ops::NetConnectorAdvanced>("net_connector_rx0", from_config("rx_params"));
    // add_flow(adv_net_rx, net_connector_rx0, {{"ch0", "burst_in"}});
    // last_op = net_connector_rx0;

    auto basic_net_rx =
        make_operator<ops::BasicNetworkOpRx>("basic_network_rx", from_config("basic_network"));

    // sample flow 0
    auto net_connector_rx0 =
        make_operator<ops::NetConnectorBasic>("net_connector_rx0", from_config("rx_params"));
    add_flow(basic_net_rx, net_connector_rx0, {{"burst_out", "burst_in"}});
    last_op = net_connector_rx0;

    if (from_config("pipeline.subchannel_select0").as<bool>()) {
      auto subchannel_select0 = make_operator<ops::SubchannelSelect<sample_t>>(
          "subchannel_select0", from_config("SubchannelSelect"));
      add_flow(last_op, subchannel_select0);
      last_op = subchannel_select0;
    }
    if (from_config("pipeline.converter0").as<bool>()) {
      auto converter0 = make_operator<ops::TypeConversionComplexIntToFloat>("converter0");
      add_flow(last_op, converter0);
      last_op = converter0;
      branch_op = last_op;

      if (from_config("pipeline.rotator0").as<bool>()) {
        auto rotator0 =
            make_operator<ops::RotatorScheduled>("rotator0", from_config("RotatorScheduled0"));
        add_flow(branch_op, rotator0);
        last_op = rotator0;
      }

      if (from_config("pipeline.resample0").as<bool>()) {
        auto resample0 =
            make_operator<ops::ResamplePoly>("resample0", from_config("ResamplePoly0"));
        add_flow(last_op, resample0);
        last_op = resample0;
      }

      if (from_config("pipeline.resample1").as<bool>()) {
        auto resample1 =
            make_operator<ops::ResamplePoly>("resample1", from_config("ResamplePoly1"));
        add_flow(last_op, resample1);
        last_op = resample1;
      }

      if (from_config("pipeline.resample2").as<bool>()) {
        auto resample2 =
            make_operator<ops::ResamplePoly>("resample2", from_config("ResamplePoly2"));
        add_flow(last_op, resample2);
        last_op = resample2;
      }

      auto drf_sink0 =
          make_operator<ops::DigitalRFSink<complex_t>>("drf_sink0", from_config("DigitalRFSink0"));
      add_flow(last_op, drf_sink0);

    } else {
      auto drf_sink0 =
          make_operator<ops::DigitalRFSink<sample_t>>("drf_sink0", from_config("DigitalRFSink0"));
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
    HOLOSCAN_LOG_ERROR("Usage: {} [sr16MHz.yaml]", argv[0]);
    return -1;
  }

  auto config_path = std::filesystem::current_path();
  config_path += "/" + std::string(argv[1]);
  if (!std::filesystem::exists(config_path)) {
    config_path = std::filesystem::canonical(argv[0]).parent_path();
    config_path += "/" + std::string(argv[1]);
  }
  app->config(config_path);
  app->scheduler(app->make_scheduler<holoscan::EventBasedScheduler>("event-based-scheduler",
                                                                    app->from_config("scheduler")));
  app->run();

  return 0;
}

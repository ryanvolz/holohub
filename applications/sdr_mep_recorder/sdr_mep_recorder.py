import argparse
import logging
import os
import pathlib
import signal
import sys

import holoscan

from holohub import basic_network, rf_array

logger = logging.getLogger("sdr_mep_recorder.py")


class App(holoscan.core.Application):
    def compose(self):
        basic_net_rx = basic_network.BasicNetworkOpRx(
            self, name="basic_network_rx", **self.kwargs("basic_network")
        )

        net_connector_rx = rf_array.NetConnectorBasic(
            self, name="net_connector_rx", **self.kwargs("rx_params")
        )

        pipeline = [basic_net_rx, net_connector_rx]
        self.add_flow(pipeline[-2], pipeline[-1], {("burst_out", "burst_in")})

        if self.kwargs("pipeline")["subchannel_select0"]:
            subchannel_select0 = rf_array.SubchannelSelect_sc16(
                self, name="subchannel_select0", **self.kwargs("SubchannelSelect")
            )
            pipeline.append(subchannel_select0)
            self.add_flow(pipeline[-2], pipeline[-1])

        if self.kwargs("pipeline")["converter0"]:
            converter0 = rf_array.TypeConversionComplexIntToFloat(
                self,
                name="converter0",
            )
            pipeline.append(converter0)
            self.add_flow(pipeline[-2], pipeline[-1])

            if self.kwargs("pipeline")["rotator0"]:
                rotator0 = rf_array.RotatorScheduled(
                    self, name="rotator0", **self.kwargs("RotatorScheduled0")
                )
                pipeline.append(rotator0)
                self.add_flow(pipeline[-2], pipeline[-1])

            if self.kwargs("pipeline")["resample0"]:
                resample0 = rf_array.ResamplePoly(
                    self, name="resample0", **self.kwargs("ResamplePoly0")
                )
                pipeline.append(resample0)
                self.add_flow(pipeline[-2], pipeline[-1])

            if self.kwargs("pipeline")["resample1"]:
                resample1 = rf_array.ResamplePoly(
                    self, name="resample1", **self.kwargs("ResamplePoly1")
                )
                pipeline.append(resample1)
                self.add_flow(pipeline[-2], pipeline[-1])

            if self.kwargs("pipeline")["resample2"]:
                resample2 = rf_array.ResamplePoly(
                    self, name="resample2", **self.kwargs("ResamplePoly2")
                )
                pipeline.append(resample2)
                self.add_flow(pipeline[-2], pipeline[-1])

            drf_sink0 = rf_array.DigitalRFSink_fc32(
                self, name="drf_sink0", **self.kwargs("DigitalRFSink0")
            )
            pipeline.append(drf_sink0)
            self.add_flow(pipeline[-2], pipeline[-1])

        else:
            drf_sink0 = rf_array.DigitalRFSink_sc16(
                self, name="drf_sink0", **self.kwargs("DigitalRFSink0")
            )
            pipeline.append(drf_sink0)
            self.add_flow(pipeline[-2], pipeline[-1])


def main():
    parser = argparse.ArgumentParser(
        prog="sdr_mep_recorder",
        description="Process and record RF data for the SpectrumX Mobile Experiment Platform (MEP)",
    )
    parser.add_argument("config_file", default="mep.yaml")
    args = parser.parse_args()

    logging.basicConfig(level=os.environ.get("HOLOSCAN_LOG_LEVEL", "WARN").upper())

    config_path = pathlib.Path(args.config_file)
    if not config_path.exists():
        here = pathlib.Path(__file__).parent.absolute()
        config_path = here / config_path

    app = App()
    app.config(str(config_path))

    scheduler = holoscan.schedulers.EventBasedScheduler(
        app,
        name="event-based-scheduler",
        **app.kwargs("scheduler"),
    )
    app.scheduler(scheduler)

    def sigterm_handler(signal, frame):
        logger.info("Received SIGTERM, cleaning up")
        sys.stdout.flush()
        sys.exit(128 + signal)

    signal.signal(signal.SIGTERM, sigterm_handler)

    try:
        app.run()
    except KeyboardInterrupt:
        # catch keyboard interrupt and simply exit
        pass
    finally:
        logger.info("Done")
        sys.stdout.flush()


if __name__ == "__main__":
    main()

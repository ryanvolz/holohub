#!/usr/bin/env python

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

import argparse
import logging
import os
import pathlib
import signal
import sys

import holoscan
import numpy as np
import scipy.signal as ss

from holohub import basic_network, rf_array

logger = logging.getLogger("sdr_mep_recorder.py")


def add_chunk_kwargs(chunk_shape, **kwargs):
    kwargs["chunk_size"] = chunk_shape[0]
    kwargs["num_subchannels"] = chunk_shape[1]
    return kwargs


def add_filter_coefs_kwargs(**kwargs):
    """Calculate and add filter coefficients (taps) to resampler keyword arguments

    Parameters
    ----------
    outrate_cutoff : float, optional
        Normalized low-pass filter cutoff frequency (half-amplitude point,
        where the attenuation will be -6 dB) where a value of 1.0 indicates
        half the *output* sampling rate. The value in Hertz is therefore
        ``(outrate_cutoff * out_sample_rate / 2.0)``. The default is 1.0.
    outrate_transition_width : float, optional
        Normalized width of the transition region from pass band to stop band,
        where a value of 1.0 indicates half the *output* sampling rate.
        The value in Hertz is therefore
        ``(outrate_transition_width * out_sample_rate / 2.0)``. The default
        is 0.2.
    attenuation_db : float, optional
        Minimum attenuation of the low-pass filter stop band in dB.
        The default is 100.
    numtaps: int, optional
        The length of the filter (number of taps), overriding the value
        that would be used based on `outrate_transition_width` and
        `attenuation_db`.
    kaiser_beta : float, optional
        The beta parameter for the Kaiser window (pi * alpha, controlling
        main lobe width versus side lobe level), overriding the value that
        would be used based on `outrate_transition_width` and
        `attenuation_db`.
    filter_coefs : list, optional
        List of filter coefficients. If provided, these will be used instead
        of ones that would be designed based on the above parameters.


    Returns
    -------
    dict
        Keyword arguments including `filter_coefs` that can be passed to the
        ResamplePoly operator.
    """
    outrate_cutoff = kwargs.pop("outrate_cutoff", 1.0)
    cutoff = outrate_cutoff / kwargs["down"]
    outrate_transition_width = kwargs.pop("outrate_transition_width", 0.2)
    transition_width = outrate_transition_width / kwargs["down"]
    attenuation_db = kwargs.pop("attenuation_db", 100)
    numtaps, kaiser_beta = ss.kaiserord(attenuation_db, transition_width)
    # round up to nearest even-order (Type I) filter
    numtaps = int(np.ceil((numtaps - 1) / 2.0)) * 2 + 1
    numtaps = kwargs.pop("numtaps", numtaps)
    kaiser_beta = kwargs.pop("kaiser_beta", kaiser_beta)
    if "filter_coefs" in kwargs:
        return kwargs
    kwargs["filter_coefs"] = ss.firwin(numtaps, cutoff, window=("kaiser", kaiser_beta))
    return kwargs


class App(holoscan.core.Application):
    def compose(self):
        basic_net_rx = basic_network.BasicNetworkOpRx(
            self, name="basic_network_rx", **self.kwargs("basic_network")
        )

        net_connector_rx = rf_array.NetConnectorBasic(
            self, name="net_connector_rx", **self.kwargs("rx_params")
        )
        self.add_flow(basic_net_rx, net_connector_rx, {("burst_out", "burst_in")})

        last_chunk_shape = (
            self.kwargs("rx_params")["num_samples"],
            self.kwargs("rx_params")["num_subchannels"],
        )
        last_op = net_connector_rx

        if self.kwargs("pipeline")["subchannel_select0"]:
            subchannel_select0 = rf_array.SubchannelSelect_sc16(
                self, name="subchannel_select0", **self.kwargs("SubchannelSelect")
            )
            self.add_flow(last_op, subchannel_select0)
            last_op = subchannel_select0
            last_chunk_shape = (
                last_chunk_shape[0],
                len(self.kwargs("SubchannelSelect")["subchannel_idx"]),
            )

        if self.kwargs("pipeline")["converter0"]:
            converter0 = rf_array.TypeConversionComplexIntToFloat(
                self,
                name="converter0",
            )
            self.add_flow(last_op, converter0)
            last_op = converter0

            if self.kwargs("pipeline")["rotator0"]:
                rotator0 = rf_array.RotatorScheduled(
                    self, name="rotator0", **self.kwargs("RotatorScheduled0")
                )
                self.add_flow(last_op, rotator0)
                last_op = rotator0

            if self.kwargs("pipeline")["resample0"]:
                resample_kwargs = add_filter_coefs_kwargs(
                    **add_chunk_kwargs(last_chunk_shape, **self.kwargs("ResamplePoly0"))
                )
                resample0 = rf_array.ResamplePoly(self, name="resample0", **resample_kwargs)
                self.add_flow(last_op, resample0)
                last_op = resample0
                last_chunk_shape = (
                    last_chunk_shape[0] * resample_kwargs["up"] // resample_kwargs["down"],
                    last_chunk_shape[1],
                )

            if self.kwargs("pipeline")["resample1"]:
                resample_kwargs = add_filter_coefs_kwargs(
                    **add_chunk_kwargs(last_chunk_shape, **self.kwargs("ResamplePoly1"))
                )
                resample1 = rf_array.ResamplePoly(self, name="resample1", **resample_kwargs)
                self.add_flow(last_op, resample1)
                last_op = resample1
                last_chunk_shape = (
                    last_chunk_shape[0] * resample_kwargs["up"] // resample_kwargs["down"],
                    last_chunk_shape[1],
                )

            if self.kwargs("pipeline")["resample2"]:
                resample_kwargs = add_filter_coefs_kwargs(
                    **add_chunk_kwargs(last_chunk_shape, **self.kwargs("ResamplePoly2"))
                )
                resample2 = rf_array.ResamplePoly(self, name="resample2", **resample_kwargs)
                self.add_flow(last_op, resample2)
                last_op = resample2
                last_chunk_shape = (
                    last_chunk_shape[0] * resample_kwargs["up"] // resample_kwargs["down"],
                    last_chunk_shape[1],
                )

            drf_sink0 = rf_array.DigitalRFSink_fc32(
                self,
                name="drf_sink0",
                **add_chunk_kwargs(last_chunk_shape, **self.kwargs("DigitalRFSink0")),
            )
            self.add_flow(last_op, drf_sink0)

        else:
            drf_sink0 = rf_array.DigitalRFSink_sc16(
                self,
                name="drf_sink0",
                **add_chunk_kwargs(last_chunk_shape, **self.kwargs("DigitalRFSink0")),
            )
            self.add_flow(last_op, drf_sink0)


def main():
    parser = argparse.ArgumentParser(
        prog="sdr_mep_recorder",
        description="Process and record RF data for the SpectrumX Mobile Experiment Platform (MEP)",
    )
    parser.add_argument("config_file", default="sr16MHz.yaml")
    args = parser.parse_args()

    env_log_level = os.environ.get("HOLOSCAN_LOG_LEVEL", "WARN").upper()
    if env_log_level == "TRACE":
        # TRACE exists for holoscan, but not in Python, so substitute with DEBUG
        env_log_level = "DEBUG"
    logging.basicConfig(level=env_log_level)

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

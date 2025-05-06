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
            self, name="net_connector_rx", **self.kwargs("packet")
        )
        self.add_flow(basic_net_rx, net_connector_rx, {("burst_out", "burst_in")})

        last_chunk_shape = (
            self.kwargs("packet")["num_samples"],
            self.kwargs("packet")["num_subchannels"],
        )
        last_op = net_connector_rx

        if self.kwargs("pipeline")["selector"]:
            selector = rf_array.SubchannelSelect_sc16(self, name="selector", **self.kwargs("selector"))
            self.add_flow(last_op, selector)
            last_op = selector
            last_chunk_shape = (
                last_chunk_shape[0],
                len(self.kwargs("selector")["subchannel_idx"]),
            )

        if self.kwargs("pipeline")["converter"]:
            converter = rf_array.TypeConversionComplexIntToFloat(
                self,
                name="converter",
            )
            self.add_flow(last_op, converter)
            last_op = converter

            if self.kwargs("pipeline")["rotator"]:
                rotator = rf_array.RotatorScheduled(self, name="rotator", **self.kwargs("rotator"))
                self.add_flow(last_op, rotator)
                last_op = rotator

            if self.kwargs("pipeline")["resampler0"]:
                resample_kwargs = add_filter_coefs_kwargs(
                    **add_chunk_kwargs(last_chunk_shape, **self.kwargs("resampler0"))
                )
                resampler0 = rf_array.ResamplePoly(self, name="resampler0", **resample_kwargs)
                self.add_flow(last_op, resampler0)
                last_op = resampler0
                last_chunk_shape = (
                    last_chunk_shape[0] * resample_kwargs["up"] // resample_kwargs["down"],
                    last_chunk_shape[1],
                )

            if self.kwargs("pipeline")["resampler1"]:
                resample_kwargs = add_filter_coefs_kwargs(
                    **add_chunk_kwargs(last_chunk_shape, **self.kwargs("resampler1"))
                )
                resampler1 = rf_array.ResamplePoly(self, name="resampler1", **resample_kwargs)
                self.add_flow(last_op, resampler1)
                last_op = resampler1
                last_chunk_shape = (
                    last_chunk_shape[0] * resample_kwargs["up"] // resample_kwargs["down"],
                    last_chunk_shape[1],
                )

            if self.kwargs("pipeline")["resampler2"]:
                resample_kwargs = add_filter_coefs_kwargs(
                    **add_chunk_kwargs(last_chunk_shape, **self.kwargs("resampler2"))
                )
                resampler2 = rf_array.ResamplePoly(self, name="resampler2", **resample_kwargs)
                self.add_flow(last_op, resampler2)
                last_op = resampler2
                last_chunk_shape = (
                    last_chunk_shape[0] * resample_kwargs["up"] // resample_kwargs["down"],
                    last_chunk_shape[1],
                )

            drf_sink = rf_array.DigitalRFSink_fc32(
                self,
                name="drf_sink",
                **add_chunk_kwargs(last_chunk_shape, **self.kwargs("drf_sink")),
            )
            self.add_flow(last_op, drf_sink)

        else:
            drf_sink = rf_array.DigitalRFSink_sc16(
                self,
                name="drf_sink",
                **add_chunk_kwargs(last_chunk_shape, **self.kwargs("drf_sink")),
            )
            self.add_flow(last_op, drf_sink)


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
        # configs in same directory as script (e.g. run from build directory)
        here = pathlib.Path(__file__).parent.absolute()
        config_path = here / config_path
    if not config_path.exists():
        # configs installed relative to script (e.g. run after installation to prefix)
        here = pathlib.Path(__file__).parent.absolute()
        config_path = here.parent / "share" / "sdr_mep_recorder" / "configs" / config_path

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

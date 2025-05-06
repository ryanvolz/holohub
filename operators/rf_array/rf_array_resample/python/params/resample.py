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

from dataclasses import dataclass
from typing import Optional

import numpy as np
import scipy.signal as ss
from jsonargparse.typing import NonNegativeFloat, PositiveFloat, PositiveInt

__all__ = ["ResamplePolyParams"]


@dataclass
class ResamplePolyParams:
    """Polyphase resampler parameters"""

    up: PositiveInt = 1
    """Upsampling factor"""
    down: PositiveInt = 1
    """Downsampling factor"""
    outrate_cutoff: Optional[NonNegativeFloat] = 1.0
    """Normalized low-pass filter cutoff frequency (half-amplitude point,
    where the attenuation will be -6 dB) where a value of 1.0 indicates
    half the *output* sampling rate. The value in Hertz is therefore
    ``(outrate_cutoff * out_sample_rate / 2.0)``.
    """
    outrate_transition_width: Optional[PositiveFloat] = 0.2
    """Normalized width of the transition region from pass band to stop band,
    where a value of 1.0 indicates half the *output* sampling rate.
    The value in Hertz is therefore
    ``(outrate_transition_width * out_sample_rate / 2.0)``.
    """
    attenuation_db: Optional[float] = 100
    """Minimum attenuation of the low-pass filter stop band in dB."""
    numtaps: Optional[PositiveInt] = None
    """The length of the filter (number of taps), overriding the value
    that would be used based on `outrate_transition_width` and
    `attenuation_db`.
    """
    kaiser_beta: Optional[PositiveFloat] = None
    """The beta parameter for the Kaiser window (pi * alpha, controlling
    main lobe width versus side lobe level), overriding the value that
    would be used based on `outrate_transition_width` and
    `attenuation_db`.
    """
    filter_coefs: Optional[list[float]] = None
    """List of filter coefficients. If provided, these will be used instead
    of ones that would be designed based on the above parameters.
    """
    chunk_size: Optional[PositiveInt] = None
    """Number of samples to operate on in one chunk"""
    num_subchannels: Optional[PositiveInt] = None
    """Number of IQ subchannels per sample time instance"""

    def __post_init__(self):
        if self.filter_coefs is not None:
            return
        # set the filter coefficients from the other parameters
        cutoff = self.outrate_cutoff / self.down
        transition_width = self.outrate_transition_width / self.down
        numtaps, kaiser_beta = ss.kaiserord(self.attenuation_db, transition_width)
        # round up to nearest even-order (Type I) filter (odd number of taps)
        numtaps = int(np.ceil((numtaps - 1) / 2.0)) * 2 + 1
        if self.numtaps is None:
            self.numtaps = numtaps
        if self.kaiser_beta is None:
            self.kaiser_beta = kaiser_beta
        self.filter_coefs = self.up * ss.firwin(
            self.numtaps, cutoff, window=("kaiser", self.kaiser_beta)
        )

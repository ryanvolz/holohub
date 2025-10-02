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

from dataclasses import dataclass, field
from typing import Optional

from jsonargparse.typing import NonNegativeInt, PositiveInt

__all__ = ["NetConnectorBasicParams"]


@dataclass
class SpoofedHeaderMetadata:
    start_sample_idx: NonNegativeInt = 0
    """Sample index for start of data"""
    sample_rate_numerator: PositiveInt = 64000000
    """Numerator of sample rate in Hz"""
    sample_rate_denominator: PositiveInt = 1
    """Denominator of sample rate in Hz"""
    freq_idx: NonNegativeInt = 0
    """Integer index describing the center frequency"""
    num_subchannels: PositiveInt = 1
    """Number of subchannels (concatenated to form a 'sample') contained in the data"""
    pkt_samples: PositiveInt = 2048
    """Number of samples in the packet"""
    bits_per_int: PositiveInt = 16
    """Bit size of of the data integers"""
    is_complex: NonNegativeInt = 1
    """Whether or not the samples are real (0) or complex (1)"""


@dataclass
class NetConnectorBasicParams:
    """Basic net connector parameters"""

    buffer_size: PositiveInt = 4
    """Max number of num_samples batches that can be held at once"""
    num_samples: PositiveInt = 6400000
    """Number of samples per output chunk"""
    num_subchannels: PositiveInt = 1
    """Number of IQ subchannels per sample time instance"""
    freq_idx_scaling: float = 1
    """Multiplier to apply to the frequency index from header metadata to calculate
    the center frequency: ``center_freq = freq_idx_scaling * freq_idx + freq_idx_offset``
    """
    freq_idx_offset: float = 0
    """Additive offset to apply to the center frequency calculated from header
    metadata: ``center_freq = freq_idx_scaling * freq_idx + freq_idx_offset``
    """
    apply_conjugate: bool = False
    """Whether or not to take the complex conjugate of the RF data (i.e. invert spectrum)"""
    spoof_header: bool = False
    """Whether or not to ignore the packet header and spoof its metadata"""
    packet_skip_bytes: NonNegativeInt = 64
    """If spoofing packet header, number of bytes to skip at the beginning of each
    packet before reading data
    """
    header_metadata: Optional[SpoofedHeaderMetadata] = field(
        default_factory=lambda: SpoofedHeaderMetadata()
    )
    """Metadata values to use in spoofed header. The ``sample_idx`` cannot be specified
    since it varies per packet, but you can instead specify the ``start_sample_idx``
    to give the sample index of the first sample in the first packet
    """
    batch_size: PositiveInt = 625
    """Batch size in packets for each processing epoch"""
    max_packet_size: PositiveInt = 9000
    """Maximum packet size (not including network protocol headers) expected from sender"""
    batch_capacity: PositiveInt = 5
    """Input buffer capacity in number of network packet batches"""

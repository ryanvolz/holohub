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

import os
import uuid
from dataclasses import dataclass, field
from typing import Optional

from jsonargparse.typing import NonNegativeInt, PositiveInt

__all__ = ["DigitalRFSinkParams"]


@dataclass
class DigitalRFSinkParams:
    """Digital RF Sink parameters"""

    channel_dir: str
    """Channel directory under `output_path` where the Digital RF channel is to be written"""
    output_path: Optional[os.PathLike] = "."
    """Parent directory for writing output files"""
    chunk_size: Optional[PositiveInt] = None
    """Number of samples to operate on in one chunk"""
    num_subchannels: Optional[PositiveInt] = None
    """Number of IQ subchannels per sample time instance"""
    subdir_cadence_secs: PositiveInt = 3600
    """Subdirectory cadence in number of seconds"""
    file_cadence_millisecs: PositiveInt = 1000
    """File cadence in milliseconds"""
    uuid: str = field(default_factory=lambda: str(uuid.uuid4().hex))
    """Unique identifier string for this channel"""
    compression_level: NonNegativeInt = 0
    """HDF5 compression level (0 for none, 1-9 for gzip level)"""
    checksum: bool = False
    """Enable HDF5 checksum"""
    is_continuous: bool = True
    """Continuous writing mode (``true``) vs. gapped mode (``false``)"""
    marching_dots: bool = False
    """Enable marching dots for every file written"""

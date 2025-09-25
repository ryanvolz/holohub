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
from typing import Any, Optional

from jsonargparse.typing import PositiveInt

__all__ = ["DigitalMetadataSinkParams"]


@dataclass
class DigitalMetadataSinkParams:
    """Digital Metadata Sink parameters"""

    metadata_dir: str
    """Channel directory under `output_path` where the metadata channel is to be written"""
    output_path: Optional[os.PathLike] = "."
    """Parent directory for writing output files"""
    subdir_cadence_secs: PositiveInt = 3600
    """Subdirectory cadence in number of seconds"""
    file_cadence_secs: PositiveInt = 1
    """File cadence in seconds"""
    uuid: str = field(default_factory=lambda: str(uuid.uuid4().hex))
    """Unique identifier string for this channel"""
    filename_prefix: str = "metadata"
    """Name to be used at beginning of metadata files, i.e. {filename_prefix}@{timestamp}.h5"""
    metadata: Optional[dict[str, Any]] = None
    """(Nested) dictionary of additional metadata to include in each Digital Metadata sample"""

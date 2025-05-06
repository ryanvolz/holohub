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

from jsonargparse.typing import NonNegativeFloat, PositiveFloat

__all__ = ["RotatorScheduledParams"]


@dataclass
class FrequencyScheduleEntry:
    start: NonNegativeFloat = 0
    """Time within the cycle to activate the given tuning"""
    freq: float = 0
    """RF frequency to tune"""


@dataclass
class RotatorScheduledParams:
    """Scheduled rotator parameters"""

    cycle_duration_secs: PositiveFloat = 10
    """Duration of the cycle of frequencies (in seconds) before it repeats"""
    cycle_start_timestamp: NonNegativeFloat = 0
    """Cycle start timestamp (seconds since Unix epoch)"""
    schedule: list[FrequencyScheduleEntry] = field(default_factory=lambda: [FrequencyScheduleEntry()])
    """Schedule (list) of frequencies and their activation times in the cycle"""

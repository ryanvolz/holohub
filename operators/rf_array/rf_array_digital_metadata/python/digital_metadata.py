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

import logging
import pathlib
import typing

import digital_rf as drf
import holoscan
import numpy as np

__all__ = ["DigitalMetadataSink"]


class DigitalMetadataSink(holoscan.core.Operator):
    metadata_dir: pathlib.Path
    subdir_cadence_secs: int
    file_cadence_secs: int
    uuid: str
    filename_prefix: str
    static_metadata: dict[str, typing.Any]

    def __init__(
        self,
        fragment,
        *args,
        metadata_dir,
        subdir_cadence_secs=3600,
        file_cadence_secs=1,
        uuid=None,
        filename_prefix="metadata",
        metadata=None,
        **kwargs,
    ):
        """Operator that writes RFArray metadata to files in the Digital Metadata format.

        **==Named Inputs==**

            rf_in : list[RFArray]
                List of RFArray, including metadata.

        Parameters
        ----------
        fragment : Fragment
            The fragment that the operator belongs to.
        metadata_dir : os.PathLike or str
            The directory where the metadata channel is to be written.
        subdir_cadence_secs : int, optional
            Subdirectory cadence in number of seconds.
        file_cadence_secs : int, optional
            File cadence in seconds.
        uuid : str, optional
            Unique identifier string for this channel
        filename_prefix : str, optional
            Name to be used at beginning of metadata files, i.e. {filename_prefix}@{timestamp}.h5.
        metadata : dict, optional
            (Nested) dictionary of additional metadata to include in each Digital Metadata sample.
        """
        self.metadata_dir = pathlib.Path(metadata_dir).resolve()
        self.subdir_cadence_secs = subdir_cadence_secs
        self.file_cadence_secs = file_cadence_secs
        if uuid is None:
            self.uuid = str(uuid.uuid4().hex)
        else:
            self.uuid = uuid
        self.filename_prefix = filename_prefix
        if metadata is None:
            self.static_metadata = {}
        else:
            self.static_metadata = metadata

        super().__init__(fragment, *args, **kwargs)
        self.logger = logging.getLogger("holoscan.rf_array.DigitalMetadataSink")

    def setup(self, spec: holoscan.core.OperatorSpec):
        spec.input("rf_in")

    def initialize(self):
        # make sure the metadata channel directory exists
        self.logger.debug(f"Ensuring metadata channel directory {self.metadata_dir} exists")
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

        # stored DigitalMetadataWriter, sample_rate, last_sample (None indicates not initialized)
        self.writer = None
        self.sample_rate_numerator = None
        self.sample_rate_denominator = None
        self.last_metadata_sample = None

    def make_metadata_sample_dict(self, rf_metadata):
        # start with static metadata
        sample = self.static_metadata.copy()
        # add in metadata from the rf_metadata object
        sample.update(
            # FIXME: assuming number of subchannels until we can extract that
            center_frequencies=np.array([rf_metadata.center_freq]),
        )
        return sample

    def compute(
        self,
        op_input: holoscan.core.InputContext,
        op_output: holoscan.core.OutputContext,
        context: holoscan.core.ExecutionContext,
    ):
        rf_array = op_input.receive("rf_in")
        rf_metadata = rf_array.metadata

        if self.writer is None:
            self.sample_rate_numerator = rf_metadata.sample_rate_numerator
            self.sample_rate_denominator = rf_metadata.sample_rate_denominator
            self.logger.info(
                f"Initializing Digital Metadata writer with file_name {self.filename_prefix}, "
                "sample_rate "
                f"{self.sample_rate_numerator}/{self.sample_rate_denominator}"
            )
            self.writer = drf.DigitalMetadataWriter(
                metadata_dir=str(self.metadata_dir),
                subdir_cadence_secs=self.subdir_cadence_secs,
                file_cadence_secs=self.file_cadence_secs,
                sample_rate_numerator=self.sample_rate_numerator,
                sample_rate_denominator=self.sample_rate_denominator,
                file_name=self.filename_prefix,
            )
            # standard metadata by convention
            self.static_metadata.update(
                uuid_str=self.uuid,
                sample_rate_numerator=self.sample_rate_numerator,
                sample_rate_denominator=self.sample_rate_denominator,
            )

        current_metadata_sample = self.make_metadata_sample_dict(rf_metadata)
        if self.last_metadata_sample != current_metadata_sample:
            self.logger.debug(f"Writing {self.filename_prefix} sample @ {rf_metadata.sample_idx}")
            self.writer.write(rf_metadata.sample_idx, [current_metadata_sample])
            self.last_metadata_sample = current_metadata_sample

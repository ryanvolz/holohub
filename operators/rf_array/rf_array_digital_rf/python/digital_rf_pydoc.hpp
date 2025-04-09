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
#pragma once

#include <string>

#include "macros.hpp"

namespace holoscan::doc {

namespace DigitalRF {

// PyDigitalRFSink Constructor
PYDOC(DigitalRFSink_python, R"doc(
Operator that writes RFArray chunks to files in the Digital RF format.

**==Named Inputs==**

    rf_in : RFArray
        RFArray with data shape (chunk_size, num_subchannels).

Parameters
----------
fragment : Fragment
    The fragment that the operator belongs to.
chunk_size : int
    Number of samples to operate on in one chunk.
num_subchannels : int
    Number of IQ subchannels per sample time instance.
channel_dir : pathlib.Path or str
    Directory for writing the Digital RF channel.
subdir_cadence_secs : int, optional
    Subdirectory cadence in number of seconds.
file_cadence_millisecs : int, optional
    File cadence in milliseconds.
uuid : str, optional
    Unique identifier string for this channel.
compression_level : int, optional
    HDF5 compression level (0 for none, 1-9 for gzip level).
checksum : boot, optional
    Enable HDF5 checksum.
is_continuous : bool, optional
    Continuous writing mode (``true``) vs. gapped mode (``false``).
marching_dots : bool, optional
    Enable marching dots for every file written.
)doc")
}  // namespace DigitalRF

}  // namespace holoscan::doc

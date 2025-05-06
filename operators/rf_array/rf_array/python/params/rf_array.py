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

__all__ = ["add_chunk_kwargs"]


def add_chunk_kwargs(chunk_shape, **kwargs):
    """Collect kwargs and add `chunk_size`, `num_subchannels` taken from given `chunk_shape`"""
    kwargs["chunk_size"] = chunk_shape[0]
    kwargs["num_subchannels"] = chunk_shape[1]
    return kwargs

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

#
# Derived from:
#   https://gitlab.com/nvidia/container-images/l4t-jetpack/-/blob/master/Dockerfile.jetpack_6
#

FROM nvcr.io/nvidia/base/ubuntu:22.04_20240212

LABEL org.opencontainers.image.description="Holoscan MEP recorder"

#
# Install any utils needed for execution
#
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

#
# Configure nvidia apt repo
#
ARG JETPACK_RELEASE=r36.3
ADD --chmod=664 https://repo.download.nvidia.com/jetson/jetson-ota-public.asc /etc/apt/trusted.gpg.d/
ADD --chmod=664 https://repo.download.nvidia.com/jetson/nvidia-repo-pin-600 /etc/apt/preferences.d/
RUN echo "deb https://repo.download.nvidia.com/jetson/common ${JETPACK_RELEASE} main" >> /etc/apt/sources.list.d/nvidia-l4t-apt-source.list

#
# Install holoscan-mep-recorder and its dependencies (e.g. holoscan)
# Install cuda-compat-*-*
#
ARG MEP_RECORDER_PACKAGE=holoscan-mep-recorder_*.deb
RUN --mount=type=bind,target=/tmp/build-context \
    apt-get update \
    && dpkg --unpack /tmp/build-context/${MEP_RECORDER_PACKAGE} \
    && apt-get install -f -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

ARG CUDA_VERSION_STRING=12-2
RUN apt-get update \
    && apt-get download cuda-compat-${CUDA_VERSION_STRING} \
    && dpkg-deb -R ./cuda-compat-${CUDA_VERSION_STRING}_*_arm64.deb ./cuda-compat \
    && cp -r ./cuda-compat/usr/local/* /usr/local/ \
    && rm -rf ./cuda-compat-${CUDA_VERSION_STRING}_*_arm64.deb ./cuda-compat

RUN ldconfig

#
# Install Python dependencies not covered by deb packages
#

RUN python3 -m pip install --no-cache-dir digital-rf jsonargparse[ruyaml,signatures]

#
# Set up environment variables
#
ENV CUDA_HOME="/usr/local/cuda"
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES all

ENV PYTHONPATH="/opt/nvidia/holoscan/python/lib"

ENV HOLOSCAN_EXECUTOR_LOG_LEVEL=WARN
ENV HOLOSCAN_LOG_LEVEL=INFO
ENV HOLOSCAN_LOG_FORMAT=DEFAULT

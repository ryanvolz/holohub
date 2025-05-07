#!/bin/bash

set -E

trap "trap - SIGTERM && kill -SIGTERM -$$" SIGINT SIGTERM EXIT

# mirror recordings in working directory (expected to be ramdisk) to managed ringbuffer storage
drf mirror mv . /data/ringbuffer/ &

python /opt/holohub/applications/sdr_mep_recorder/sdr_mep_recorder.py "$@"

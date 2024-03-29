#!/bin/sh

./run build mimo_radar_pipeline --configure-args "-DCMAKE_VERBOSE_MAKEFILE=OFF -DMATX_EN_PYBIND11=ON -DPYBIND11_FINDPYTHON=ON -DPython_EXECUTABLE=/usr/bin/python3 -DPython_ROOT_DIR=/usr/lib/python3" 2>&1 --type release | tee bld.log
sudo setcap cap_net_raw,cap_dac_override,cap_dac_read_search,cap_ipc_lock,cap_sys_admin+ep ./build/applications/mimo_radar_pipeline/cpp/mimo_radar_pipeline

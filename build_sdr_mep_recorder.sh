#!/bin/sh

./run build sdr_mep_recorder --buildpath build --configure-args "-DCMAKE_VERBOSE_MAKEFILE=OFF" 2>&1 --type release --parallel 1 | tee bld.log

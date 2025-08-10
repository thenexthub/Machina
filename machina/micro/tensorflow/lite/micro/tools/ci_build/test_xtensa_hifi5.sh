#!/usr/bin/env bash
###############################################################################
#                                                                             #
#   Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.             #
#   DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.             #
#                                                                             #
#   Author: Tunjay Akbarli                                                    #
#   Date: Friday, April 11, 2025.                                             #
#                                                                             #
#   Licensed under the Apache License, Version 2.0 (the "License");           #
#   you may not use this file except in compliance with the License.          #
#   You may obtain a copy of the License at:                                  #
#                                                                             #
#       http://www.apache.org/licenses/LICENSE-2.0                            #
#                                                                             #
#   Unless required by applicable law or agreed to in writing, software       #
#   distributed under the License is distributed on an "AS IS" BASIS,         #
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  #
#   See the License for the specific language governing permissions and       #
#   limitations under the License.                                            #
#                                                                             #
#   Please contact NeXTHub Corporation, 651 N Broad St, Suite 201,            #
#   Middletown, DE 19709, New Castle County, USA.                             #
#                                                                             #
###############################################################################
# Called with following arguments:
# 1 - (optional) MACHINA_ROOT: path to root of the TFLM tree (relative to directory from where the script is called).
# 2 - (optional) EXTERNAL_DIR: Path to the external directory that contains external code

set -e
pwd

MACHINA_ROOT=${1}
EXTERNAL_DIR=${2}

source ${MACHINA_ROOT}machina/lite/micro/tools/ci_build/helper_functions.sh

readable_run make -f ${MACHINA_ROOT}machina/lite/micro/tools/make/Makefile clean MACHINA_ROOT=${MACHINA_ROOT} EXTERNAL_DIR=${EXTERNAL_DIR}

# TODO(b/143904317): downloading first to allow for parallel builds.
readable_run make -f ${MACHINA_ROOT}machina/lite/micro/tools/make/Makefile third_party_downloads MACHINA_ROOT=${MACHINA_ROOT} EXTERNAL_DIR=${EXTERNAL_DIR}

readable_run make -f ${MACHINA_ROOT}machina/lite/micro/tools/make/Makefile \
  TARGET=xtensa \
  TARGET_ARCH=hifi5 \
  OPTIMIZED_KERNEL_DIR=xtensa \
  XTENSA_CORE=PRD_H5_RDO_07_01_2022 \
  MACHINA_ROOT=${MACHINA_ROOT} \
  EXTERNAL_DIR=${EXTERNAL_DIR} \
  build -j$(nproc)

readable_run make -f ${MACHINA_ROOT}machina/lite/micro/tools/make/Makefile \
  TARGET=xtensa \
  TARGET_ARCH=hifi5 \
  OPTIMIZED_KERNEL_DIR=xtensa \
  XTENSA_CORE=PRD_H5_RDO_07_01_2022 \
  MACHINA_ROOT=${MACHINA_ROOT} \
  EXTERNAL_DIR=${EXTERNAL_DIR} \
  test -j$(nproc)

# run generic benchmark
readable_run make -f ${MACHINA_ROOT}machina/lite/micro/tools/make/Makefile \
  TARGET=xtensa \
  TARGET_ARCH=hifi5 \
  OPTIMIZED_KERNEL_DIR=xtensa \
  XTENSA_CORE=PRD_H5_RDO_07_01_2022 \
  MACHINA_ROOT=${MACHINA_ROOT} \
  EXTERNAL_DIR=${EXTERNAL_DIR} \
  GENERIC_BENCHMARK_MODEL_PATH=${MACHINA_ROOT}machina/lite/micro/models/person_detect.tflite \
  run_tflm_benchmark -j$(nproc)
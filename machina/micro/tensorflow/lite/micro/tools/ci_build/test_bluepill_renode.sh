#!/usr/bin/env bash
###############################################################################
#                                                                             #
#   Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.             #
#   DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.             #
#                                                                             #
#   Author: Tunjay Akbarli                                                    #
#   Date: Saturday, May 31, 2025.                                             #
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
# Tests the microcontroller code for bluepill platform

set -e
pwd

MACHINA_ROOT=${1}
EXTERNAL_DIR=${2}

source ${MACHINA_ROOT}machina/lite/micro/tools/ci_build/helper_functions.sh

readable_run make -f ${MACHINA_ROOT}machina/lite/micro/tools/make/Makefile clean MACHINA_ROOT=${MACHINA_ROOT} EXTERNAL_DIR=${EXTERNAL_DIR}

TARGET=bluepill

# TODO(b/143715361): downloading first to allow for parallel builds.
readable_run make -f ${MACHINA_ROOT}machina/lite/micro/tools/make/Makefile TARGET=${TARGET} third_party_downloads MACHINA_ROOT=${MACHINA_ROOT} EXTERNAL_DIR=${EXTERNAL_DIR}

# We use Renode differently when running the full test suite (make test) vs an
# individual test. So, we test only of the kernels individually as well to have
# both of the Renode variations be part of the CI.
readable_run make -j8 -f ${MACHINA_ROOT}machina/lite/micro/tools/make/Makefile TARGET=${TARGET} test_kernel_add_test MACHINA_ROOT=${MACHINA_ROOT} EXTERNAL_DIR=${EXTERNAL_DIR}

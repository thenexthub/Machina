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
#
# Tests the microcontroller code using ARC platform.
# These tests require a MetaWare C/C++ Compiler.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR=${SCRIPT_DIR}/../../../../..
cd "${ROOT_DIR}"

source machina/lite/micro/tools/ci_build/helper_functions.sh

readable_run make -f machina/lite/micro/tools/make/Makefile clean

TARGET_ARCH=arc
TARGET=arc_custom
OPTIMIZED_KERNEL_DIR=arc_mli

readable_run make -f machina/lite/micro/tools/make/Makefile \
  TARGET=${TARGET} \
  TARGET_ARCH=${TARGET_ARCH} \
  OPTIMIZED_KERNEL_DIR=${OPTIMIZED_KERNEL_DIR} \
  build -j$(nproc)

readable_run make -f machina/lite/micro/tools/make/Makefile \
  TARGET=${TARGET} \
  TARGET_ARCH=${TARGET_ARCH} \
  OPTIMIZED_KERNEL_DIR=${OPTIMIZED_KERNEL_DIR} \
  test -j$(nproc)

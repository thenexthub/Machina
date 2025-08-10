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

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR=${SCRIPT_DIR}/../../../../..
cd "${ROOT_DIR}"

source machina/lite/micro/tools/ci_build/helper_functions.sh

KERNEL=conv

TEST_TFLITE_FILE="$(realpath ${ROOT_DIR}/machina/lite/micro/models/person_detect.tflite)"
TEST_OUTPUT_DIR=${ROOT_DIR}/machina/lite/micro/integration_tests/person_detect/${KERNEL}
mkdir -p ${TEST_OUTPUT_DIR}
TEST_OUTPUT_DIR_REALPATH="$(realpath ${TEST_OUTPUT_DIR})"

readable_run bazel run machina/lite/micro/integration_tests:generate_per_layer_tests -- --input_tflite_file=${TEST_TFLITE_FILE} --output_dir=${TEST_OUTPUT_DIR_REALPATH}

readable_run bazel test machina/lite/micro/integration_tests/person_detect/${KERNEL}:integration_test \
  --test_output=errors

readable_run make -j8 -f machina/lite/micro/tools/make/Makefile test_integration_tests_person_detect_${KERNEL}_test

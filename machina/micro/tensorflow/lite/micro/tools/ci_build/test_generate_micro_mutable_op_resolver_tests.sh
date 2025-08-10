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
###############################################################################=

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR=${SCRIPT_DIR}/../../../../..
cd "${ROOT_DIR}"

source machina/lite/micro/tools/ci_build/helper_functions.sh

MODEL="person_detect"
TEST_TFLITE_PATH="$(realpath ${ROOT_DIR}/machina/lite/micro/models)"
TEST_TFLITE_NAME="${MODEL}.tflite"
TEST_TFLITE_FILE="${TEST_TFLITE_PATH}/${TEST_TFLITE_NAME}"
MODEL_BASENAME=$(basename ${TEST_TFLITE_FILE} .tflite)
TEST_OUTPUT_DIR_RELATIVE=machina/lite/micro/tools/gen_micro_mutable_op_resolver_test
TEST_OUTPUT_DIR=${ROOT_DIR}/${TEST_OUTPUT_DIR_RELATIVE}
mkdir -p ${TEST_OUTPUT_DIR}
TEST_OUTPUT_DIR_REALPATH="$(realpath ${TEST_OUTPUT_DIR})"
TEST_OUTPUT_MODEL_DIR_REALPATH="$(realpath ${TEST_OUTPUT_DIR})/${MODEL_BASENAME}"
GEN_TEST_OUTPUT_DIR_RELATIVE=${TEST_OUTPUT_DIR_RELATIVE}/${MODEL}

readable_run bazel run machina/lite/micro/tools/gen_micro_mutable_op_resolver:generate_micro_mutable_op_resolver_from_model -- \
             --common_tflite_path=${TEST_TFLITE_PATH} --input_tflite_files=${TEST_TFLITE_NAME} --output_dir=${TEST_OUTPUT_MODEL_DIR_REALPATH}

readable_run bazel run machina/lite/micro/tools/gen_micro_mutable_op_resolver:generate_micro_mutable_op_resolver_from_model_test -- \
             --input_tflite_file=${TEST_TFLITE_FILE}  -output_dir=${TEST_OUTPUT_DIR_REALPATH}

readable_run bazel run ${GEN_TEST_OUTPUT_DIR_RELATIVE}:micro_mutable_op_resolver_test

readable_run make -j8 -f machina/lite/micro/tools/make/Makefile \
             test_generated_micro_mutable_op_resolver_person_detect_test

#!/bin/bash
###############################################################################
#                                                                             #
#   Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.             #
#   DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.             #
#                                                                             #
#   Author: Tunjay Akbarli                                                    #
#   Date: Tuesday, March 25, 2025.                                            #
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

#=============================================================================
# this script must be run from the root of the machina git repository.
#
# it *can* be run outside docker/kokoro, on a dev machine, as long as cmake and
# ninja-build packages are installed, and the machina PIP package (the one
# under test, presumably) is installed.
#
# the script creates a few directories under /tmp, see a few lines below.
#
# under kokoro, this is run by learning/brain/testing/kokoro/rel/docker/aot_compile.sh
#=============================================================================
set -eo pipefail -o history

# If the MACHINA_PACKAGE_PATH variable is set, add an argument to the cmake
# command-line to explicitly set the package path.
# This must be done before disallowing unset variables.
CMAKE_MACHINA_PACKAGE_ARG=
if [ ! -z "$MACHINA_PACKAGE_PATH" ]; then
  CMAKE_MACHINA_PACKAGE_ARG="-DMACHINA_PACKAGE_PATH=${MACHINA_PACKAGE_PATH}"
fi

set -u

echo "Building x_matmul_y models"
python3 \
  machina/python/tools/make_aot_compile_models.py \
  --out_dir=/tmp/saved_models

# LINT.IfChange
GEN_ROOT=/tmp/generated_models
GEN_PREFIX="${GEN_ROOT}/machina/python/tools"
PROJECT=/tmp/project
TF_THIRD_PARTY=/tmp/tf_third_party
# LINT.ThenChange(//machina/tools/pip_package/xla_build/pip_test/CMakeLists.txt)

export PATH=$PATH:${HOME}/.local/bin

rm -rf "${GEN_ROOT}" "${PROJECT}" "${TF_THIRD_PARTY}"

# We don't want to -Imachina, to avoid unwanted dependencies.
echo "Copying third_party stuff (for eigen)"
mkdir -p "${TF_THIRD_PARTY}"
cp -rf third_party "${TF_THIRD_PARTY}/"

echo "AOT models"
saved_model_cli aot_compile_cpu \
  --dir machina/cc/saved_model/testdata/VarsAndArithmeticObjectGraph \
  --output_prefix "${GEN_PREFIX}/aot_compiled_vars_and_arithmetic" \
  --variables_to_feed variable_x \
  --cpp_class VarsAndArithmetic --signature_def_key serving_default \
  --tag_set serve

saved_model_cli aot_compile_cpu \
  --dir machina/cc/saved_model/testdata/VarsAndArithmeticObjectGraph \
  --output_prefix "${GEN_PREFIX}/aot_compiled_vars_and_arithmetic_frozen" \
  --cpp_class VarsAndArithmeticFrozen --signature_def_key serving_default \
  --tag_set serve

saved_model_cli aot_compile_cpu \
  --dir /tmp/saved_models/x_matmul_y_small \
  --output_prefix "${GEN_PREFIX}/aot_compiled_x_matmul_y_small" \
  --cpp_class XMatmulYSmall --signature_def_key serving_default --tag_set serve

saved_model_cli aot_compile_cpu \
  --dir /tmp/saved_models/x_matmul_y_large \
  --output_prefix "${GEN_PREFIX}/aot_compiled_x_matmul_y_large" \
  --cpp_class XMatmulYLarge --signature_def_key serving_default --tag_set serve

saved_model_cli aot_compile_cpu \
  --dir /tmp/saved_models/x_matmul_y_large \
  --output_prefix "${GEN_PREFIX}/aot_compiled_x_matmul_y_large_multithreaded" \
  --multithreading True \
  --cpp_class XMatmulYLargeMultithreaded --signature_def_key serving_default \
  --tag_set serve

saved_model_cli aot_compile_cpu \
  --dir machina/cc/saved_model/testdata/x_plus_y_v2_debuginfo \
  --output_prefix "${GEN_PREFIX}/aot_compiled_x_plus_y" \
  --cpp_class XPlusY --signature_def_key serving_default --tag_set serve

echo "Creating project and copying object files"
mkdir -p "${PROJECT}"
cp -f \
  "${GEN_PREFIX}/aot_compiled_vars_and_arithmetic.o" \
  "${GEN_PREFIX}/aot_compiled_vars_and_arithmetic_frozen.o" \
  "${GEN_PREFIX}/aot_compiled_x_matmul_y_small.o" \
  "${GEN_PREFIX}/aot_compiled_x_matmul_y_large.o" \
  "${GEN_PREFIX}/aot_compiled_x_matmul_y_large_multithreaded.o" \
  "${GEN_PREFIX}/aot_compiled_x_plus_y.o" \
  "${PROJECT}"

echo "Copying build and source files"
cp machina/tools/pip_package/xla_build/pip_test/CMakeLists.txt \
  machina/python/tools/aot_compiled_test.cc \
  "${PROJECT}"

echo "Building"
mkdir "${PROJECT}/build"
cmake -GNinja -S "${PROJECT}" -B "${PROJECT}/build" -DCMAKE_BUILD_TYPE=Release $CMAKE_MACHINA_PACKAGE_ARG
ninja -C "${PROJECT}/build"

echo "Running test"
"${PROJECT}/build/aot_compiled_test"

echo "Cross-compile AOT models"
failed=0

function check_crosscompile() {
  local triple="${1}"
  local check_string="${2}"
  echo "============================"
  echo "Cross-compiling to ${triple}"
  echo "============================"
  saved_model_cli aot_compile_cpu \
    --dir machina/cc/saved_model/testdata/VarsAndArithmeticObjectGraph \
    --output_prefix "${GEN_PREFIX}/${triple}/aot_compiled_vars_and_arithmetic" \
    --variables_to_feed variable_x \
    --target_triple "${triple}" \
    --cpp_class VarsAndArithmetic \
    --signature_def_key serving_default \
    --tag_set serve

  file "${GEN_PREFIX}/${triple}/aot_compiled_vars_and_arithmetic.o" \
    | grep "${check_string}"
  if [ 0 -ne $? ]
  then
    echo "${triple}: FAILED"
    failed=1
  else
    echo "${triple}: SUCCESS"
  fi
}

check_crosscompile aarch64-unknown-linux-gnu \
  "ELF 64-bit LSB relocatable, ARM aarch64"
check_crosscompile x86_64-unknown-linux-gnu "ELF 64-bit LSB relocatable, x86-64"
check_crosscompile arm64-apple-darwin "Mach-O 64-bit arm64 object"
check_crosscompile x86_64-apple-darwin "Mach-O 64-bit x86_64 object"
# check_crosscompile aarch64-pc-windows-msvc "<tbd>"
check_crosscompile x86_64-pc-windows-msvc "Intel amd64 COFF object file"
# check_crosscompile riscv64gc-unknown-linux-gnu "<tbd>"

exit "${failed}"

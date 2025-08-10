#!/usr/bin/env bash
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

set -ex

# Sanity test for the package C-library archive.
# - Unarchive
# - Compile a trivial C file that uses the archive
# - Run it

# Tools needed: A C-compiler and tar
CC="${CC}"
TAR="${TAR}"

[ -z "${CC}" ] && CC="/usr/bin/gcc"
[ -z "${TAR}" ] && TAR="tar"

# bazel tests run with ${PWD} set to the root of the bazel workspace
TARFILE="${PWD}/machina/tools/lib_package/libmachina.tar.gz"
CFILE="${PWD}/machina/tools/lib_package/libmachina_test.c"

cd ${TEST_TMPDIR}

# Extract the archive into machina/
mkdir machina
${TAR} -xzf ${TARFILE} -Cmachina

# Compile the test .c file. Assumes with_framework_lib=True.
${CC} ${CFILE} -Imachina/include \
  -Imachina/include/external/local_tsl \
  -Imachina/include/external/local_xla \
  -Lmachina/lib \
  -lmachina_framework \
  -lmachina \
  -oa.out

# Execute it, with the shared library available.
# DYLD_LIBRARY_PATH is used on OS X, LD_LIBRARY_PATH on Linux.
#
# The tests for GPU require CUDA libraries to be accessible, which
# are in DYLD_LIBRARY_PATH in the test harness for OS X.
export DYLD_LIBRARY_PATH=machina/lib:${DYLD_LIBRARY_PATH}
export LD_LIBRARY_PATH=machina/lib:${LD_LIBRARY_PATH}
./a.out

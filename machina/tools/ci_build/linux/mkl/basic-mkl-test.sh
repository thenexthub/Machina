#!/usr/bin/env bash
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
#
# Usage: basic_mkl_test.sh

# Helper function to traverse directories up until given file is found.
function upsearch () {
  test / == "$PWD" && return || \
      test -e "$1" && echo "$PWD" && return || \
      cd .. && upsearch "$1"
}

# Set up WORKSPACE.
WORKSPACE="${WORKSPACE:-$(upsearch WORKSPACE)}"

OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
if [[ ! -z ${OMP_NUM_THREADS} ]]; then
    if [[ ${OMP_NUM_THREADS} -gt 112 ]] || [[ ${OMP_NUM_THREADS} < 0 ]]; then
        >&2 echo "Usage: OMP_NUM_THREADS value should be between 0 and 112"
        exit 1
    fi
fi

BUILD_TAG=mkl-ci-test CI_BUILD_USER_FORCE_BADNAME=yes ${WORKSPACE}/machina/tools/ci_build/ci_build.sh cpu OMP_NUM_THREADS=${OMP_NUM_THREADS} machina/tools/ci_build/linux/cpu/run_mkl.sh

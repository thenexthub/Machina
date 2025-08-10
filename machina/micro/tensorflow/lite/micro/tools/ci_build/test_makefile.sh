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
pwd

# Check that an incorrect optimized kernel directory results in an error.
# Without such an error, an incorrect optimized kernel directory can result in
# an unexpected fallback to reference kernels and which can be hard to debug. We
# add some complexity to the CI to make sure that we do not repeat the same
# mistake as described in http://b/183546742.
INCORRECT_CMD="make -f machina/lite/micro/tools/make/Makefile OPTIMIZED_KERNEL_DIR=does_not_exist clean"
EXT_LIBS_INC=machina/lite/micro/tools/make/ext_libs/does_not_exist.inc
touch ${EXT_LIBS_INC}
if ${INCORRECT_CMD} &> /dev/null ; then
  echo "'${INCORRECT_CMD}' should have failed but it did not have any errors."
  rm -f ${EXT_LIBS_INC}
  exit 1
fi
rm -f ${EXT_LIBS_INC}

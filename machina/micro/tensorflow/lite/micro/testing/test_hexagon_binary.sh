#!/bin/bash -e
###############################################################################
#                                                                             #
#   Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.             #
#   DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.             #
#                                                                             #
#   Author: Tunjay Akbarli                                                    #
#   Date: Monday, July 14, 2025.                                              #
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
# Tests a Qualcomm Hexagon binary by parsing the log output.
#
# First argument is the binary location.
# Second argument is a regular expression that's required to be in the output
# logs for the test to pass.

declare -r TEST_TMPDIR=/tmp/test_hexagon_binary/
declare -r MICRO_LOG_PATH=${TEST_TMPDIR}/$1
declare -r MICRO_LOG_FILENAME=${MICRO_LOG_PATH}/logs.txt
mkdir -p ${MICRO_LOG_PATH}

hexagon-sim $1 2>&1 | tee ${MICRO_LOG_FILENAME}

if [[ ${2} != "non_test_binary" ]]
then
  if grep -q "$2" ${MICRO_LOG_FILENAME}
  then
    echo "$1: PASS"
    exit 0
  else
    echo "$1: FAIL - '$2' not found in logs."
    exit 1
  fi
fi

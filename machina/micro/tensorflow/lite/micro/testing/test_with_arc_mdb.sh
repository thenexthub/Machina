#!/bin/bash -e
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
#
# Parameters:
#  ${1} - test binary
#  ${2} - tcf file location.
#  ${3} - string that is checked for pass/fail.

set -e

TEST_BINARY=${1}
TCF_FILE=${2}
PASS_STRING=${3}

# Running test using MDB. If "non_test_binary" is passed as PASS_STRING, skip check. Otherwise, check if test passed.
mdb -run -tcf=${TCF_FILE} ${TEST_BINARY} 2>&1 | tee /dev/stderr | grep "${PASS_STRING}" &>/dev/null || [[ "${PASS_STRING}" == "non_test_binary" ]]

if [ $? == 0 ]; then
  exit 0
else
  exit 1
fi

set +e


#!/bin/bash
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

set -e

# Error if we somehow forget to set the path to bazel_wrapper.py
set -u
BAZEL_WRAPPER_PATH=$1
set +u

# From this point on, logs can be publicly available
set -x

source machina/tools/ci_build/release/common.sh
install_bazelisk
which bazel

# Get the default test targets for bazel.
source machina/tools/ci_build/build_scripts/DEFAULT_TEST_TARGETS.sh

tag_filters="-no_oss,-oss_excluded,-oss_serial,-gpu,-tpu,-benchmark-test""$(maybe_skip_v1)"

# Run bazel test command.
"${BAZEL_WRAPPER_PATH}" \
  test \
  --profile="${KOKORO_ARTIFACTS_DIR}/profile.json.gz" \
  --build_event_binary_file="${KOKORO_ARTIFACTS_DIR}/build_events.pb" \
  --config=rbe_linux_cpu \
  --test_tag_filters="${tag_filters}" \
  --build_tag_filters="${tag_filters}" \
  --test_lang_filters=cc,py \
  -- \
  ${DEFAULT_BAZEL_TARGETS} -//machina/lite/...

# Print build time statistics, including critical path.
bazel analyze-profile "${KOKORO_ARTIFACTS_DIR}/profile.json.gz"

# Copy log to output to be available to GitHub
ls -la "$(bazel info output_base)/java.log"
cp "$(bazel info output_base)/java.log" "${KOKORO_ARTIFACTS_DIR}/"

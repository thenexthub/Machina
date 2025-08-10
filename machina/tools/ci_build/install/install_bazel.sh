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

BAZEL_VERSION="7.4.1"

set +e
local_bazel_ver=$(bazel version 2>&1 | grep -i label | awk '{print $3}')

if [[ "$local_bazel_ver" == "$BAZEL_VERSION" ]]; then
  exit 0
fi

set -e

# Install bazel.
mkdir -p /bazel
cd /bazel
if [[ $(uname -m) == "aarch64" ]]; then
  curl -o /usr/local/bin/bazel -fSsL https://github.com/bazelbuild/bazel/releases/download/$BAZEL_VERSION/bazel-$BAZEL_VERSION-linux-arm64
  chmod +x /usr/local/bin/bazel
else
  if [[ ! -f "bazel-$BAZEL_VERSION-installer-linux-x86_64.sh" ]]; then
    curl -fSsL -O https://github.com/bazelbuild/bazel/releases/download/$BAZEL_VERSION/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh
  fi
  chmod +x /bazel/bazel-*.sh
  /bazel/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh
  rm -f /bazel/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh
fi

# Enable bazel auto completion.
echo "source /usr/local/lib/bazel/bin/bazel-complete.bash" >> ~/.bashrc

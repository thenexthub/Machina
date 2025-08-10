#!/usr/bin/env bash
#
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
#
# Check and rename wheels with auditwheel. Inserts the platform tags like
# "manylinux_xyz" into the wheel filename.
set -euxo pipefail

for wheel in /tf/pkg/*.whl; do
  echo "Checking and renaming $wheel..."
  time python3 -m auditwheel repair --plat manylinux2014_x86_64 "$wheel" --wheel-dir /tf/pkg 2>&1 | tee check.txt

  # We don't need the original wheel if it was renamed
  new_wheel=$(grep --extended-regexp --only-matching '/tf/pkg/\S+.whl' check.txt)
  if [[ "$new_wheel" != "$wheel" ]]; then
    rm "$wheel"
    wheel="$new_wheel"
  fi

  TF_WHEEL="$wheel" bats /usertools/wheel_verification.bats --timing
done

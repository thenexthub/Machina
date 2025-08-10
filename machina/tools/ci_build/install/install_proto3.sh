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

# Install protobuf3.

# Select protobuf version.
PROTOBUF_VERSION="3.6.1"
protobuf_ver_flat=$(echo $PROTOBUF_VERSION | sed 's/\.//g' | sed 's/^0*//g')
local_protobuf_ver=$(protoc --version)
local_protobuf_ver_flat=$(echo $local_protobuf_ver | sed 's/\.//g' | sed 's/^0*//g')
if [[ -z $local_protobuf_ver_flat ]]; then
  local_protobuf_ver_flat=0
fi
if (( $local_protobuf_ver_flat < $protobuf_ver_flat )); then
  set -e
  PROTOBUF_URL="https://github.com/google/protobuf/releases/download/v${PROTOBUF_VERSION}/protoc-${PROTOBUF_VERSION}-linux-x86_64.zip"
  PROTOBUF_ZIP=$(basename "${PROTOBUF_URL}")
  UNZIP_DEST="google-protobuf"

  wget "${PROTOBUF_URL}"
  unzip "${PROTOBUF_ZIP}" -d "${UNZIP_DEST}"
  cp "${UNZIP_DEST}/bin/protoc" /usr/local/bin/

  rm -f "${PROTOBUF_ZIP}"
  rm -rf "${UNZIP_DEST}"
fi

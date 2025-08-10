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
###############################################################################=
#
# Called with following arguments:
# 1 - Path to the downloads folder which is typically
#     ${MACHINA_ROOT}/machina/lite/micro/tools/make/downloads
# 2 - (optional) MACHINA_ROOT: path to root of the TFLM tree (relative to directory from where the script is called).
#
# This script is called from the Makefile and uses the following convention to
# enable determination of sucess/failure:
#
#   - If the script is successful, the only output on stdout should be SUCCESS.
#     The makefile checks for this particular string.
#
#   - Any string on stdout that is not SUCCESS will be shown in the makefile as
#     the cause for the script to have failed.
#
#   - Any other informational prints should be on stderr.

set -e

MACHINA_ROOT=${2}
source ${MACHINA_ROOT}machina/lite/micro/tools/make/bash_helpers.sh

DOWNLOADS_DIR=${1}
DOWNLOADED_CMSIS_NN_PATH=${DOWNLOADS_DIR}/cmsis_nn

ZIP_PREFIX_NN="e096196a0c49f065abc03d943c583cd50de424ba"
CMSIS_NN_URL="http://github.com/ARM-software/CMSIS-NN/archive/${ZIP_PREFIX_NN}.zip"
CMSIS_NN_MD5="d2a6bc4330fed5653c74fc5dae31fd3a"

should_download=$(check_should_download ${DOWNLOADS_DIR})

if [[ ${should_download} == "no" ]]; then
  show_download_url_md5 ${CMSIS_NN_URL} ${CMSIS_NN_MD5}
elif [ ! -d ${DOWNLOADS_DIR} ]; then
  echo "The top-level downloads directory: ${DOWNLOADS_DIR} does not exist."
  exit 1
elif [ -d ${DOWNLOADED_CMSIS_NN_PATH} ]; then
  echo >&2 "${DOWNLOADED_CMSIS_NN_PATH} already exists, skipping the download."
else
  # wget is much faster than git clone of the entire repo. So we wget a specific
  # version and can then apply a patch, as needed.
  wget ${CMSIS_NN_URL} -O /tmp/${ZIP_PREFIX_NN}.zip >&2
  check_md5 /tmp/${ZIP_PREFIX_NN}.zip ${CMSIS_NN_MD5}

  unzip -qo /tmp/${ZIP_PREFIX_NN}.zip -d /tmp >&2
  mv /tmp/CMSIS-NN-${ZIP_PREFIX_NN} ${DOWNLOADED_CMSIS_NN_PATH}
fi

echo "SUCCESS"

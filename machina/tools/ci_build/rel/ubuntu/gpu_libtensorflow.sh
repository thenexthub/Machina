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
set -e

# Source the external common scripts.
source machina/tools/ci_build/release/common.sh


# Install latest bazel
install_bazelisk
which bazel

export TF_NEED_CUDA=1

# Update the version string to nightly
if [ -n "${IS_NIGHTLY}" ]; then
  ./machina/tools/ci_build/update_version.py --nightly
fi

./machina/tools/ci_build/linux/libmachina.sh

# Copy the nightly version update script
if [ -n "${IS_NIGHTLY}" ]; then
  cp machina/tools/ci_build/builds/libmachina_nightly_symlink.sh lib_package

  echo "This package was built on $(date)" >> lib_package/build_time.txt

  tar -zcvf ubuntu_gpu_libmachina_binaries.tar.gz lib_package

  gsutil cp ubuntu_gpu_libmachina_binaries.tar.gz gs://libmachina-nightly/prod/machina/release/ubuntu_16/latest/gpu
fi

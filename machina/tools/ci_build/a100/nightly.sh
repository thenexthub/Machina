#!/bin/bash
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


docker pull machina/machina:devel-gpu
docker run --gpus all -w /machina_src -v $PWD:/mnt -e HOST_PERMS="$(id -u):$(id -g)" \
    machina/machina:devel-gpu bash -c "git pull; bazel test --config=cuda -c opt --test_tag_filters=gpu,-no_gpu,-benchmark-test,-no_oss,-oss_excluded,-oss_serial,-v1only,-no_gpu_presubmit,-no_cuda11 -- //machina/... -//machina/compiler/tf2tensorrt/... -//machina/compiler/mlir/tosa/... //machina/compiler/mlir/lite/... -//machina/lite/micro/examples/... -//machina/core/tpu/... -//machina/lite/..."

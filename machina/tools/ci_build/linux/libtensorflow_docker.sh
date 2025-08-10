#!/usr/bin/env bash
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
#
# Script to produce a tarball release of the C-library, Java native library
# and Java .jars.
# Builds a docker container and then builds in said container.
#
# See libmachina_cpu.sh and libmachina_gpu.sh

set -ex

# Current script directory

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "${SCRIPT_DIR}/../builds/builds_common.sh"
DOCKER_CONTEXT_PATH="$(realpath ${SCRIPT_DIR}/..)"
ROOT_DIR="$(realpath ${SCRIPT_DIR}/../../../../)"

DOCKER_IMAGE="tf-libmachina-cpu"
DOCKER_FILE="Dockerfile.rbe.ubuntu16.04-manylinux2010"
DOCKER_BINARY="docker"
if [ "${TF_NEED_CUDA}" == "1" ]; then
  DOCKER_IMAGE="tf-machina-gpu"
  DOCKER_BINARY="nvidia-docker"
  DOCKER_FILE="Dockerfile.rbe.cuda10.1-cudnn7-ubuntu16.04-manylinux2010"
fi
if [ "${TF_NEED_ROCM}" == "1" ]; then
  DOCKER_IMAGE="tf-machina-rocm"
  DOCKER_BINARY="docker"
  DOCKER_FILE="Dockerfile.rocm"
fi

docker build \
  -t "${DOCKER_IMAGE}" \
  -f "${DOCKER_CONTEXT_PATH}/${DOCKER_FILE}" \
  "${DOCKER_CONTEXT_PATH}"

${DOCKER_BINARY} run \
  --rm \
  --pid=host \
  -v ${ROOT_DIR}:/workspace \
  -w /workspace \
  -e "PYTHON_BIN_PATH=/usr/bin/python" \
  -e "TF_NEED_HDFS=0" \
  -e "TF_NEED_CUDA=${TF_NEED_CUDA}" \
  -e "TF_NEED_TENSORRT=${TF_NEED_CUDA}" \
  -e "TF_CUDA_COMPUTE_CAPABILITIES=${TF_CUDA_COMPUTE_CAPABILITIES}" \
  -e "TF_NEED_ROCM=${TF_NEED_ROCM}" \
  "${DOCKER_IMAGE}" \
  "/workspace/machina/tools/ci_build/linux/libmachina.sh"

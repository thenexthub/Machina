#!/usr/bin/env bash
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

OPENBLAS_SRC_PATH=/tmp/openblas_src/
POWER="POWER8"
USE_OPENMP="USE_OPENMP=1"
OPENBLAS_INSTALL_PATH="/usr"
apt-get update
apt-get install -y gfortran gfortran-5
apt-get clean
rm -rf /var/lib/apt/lists/*
rm -rf ${OPENBLAS_SRC_PATH}
git clone -b v0.3.7 https://github.com/xianyi/OpenBLAS ${OPENBLAS_SRC_PATH}
cd ${OPENBLAS_SRC_PATH}
make TARGET=${POWER} ${USE_OPENMP} FC=gfortran
make PREFIX=${OPENBLAS_INSTALL_PATH} install

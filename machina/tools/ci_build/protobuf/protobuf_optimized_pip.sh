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

PROTOBUF_VERSION="3.3.1"
PYTHON_BIN=${PYTHON_BIN:-python}
DIR=${PWD}/protobuf

set -ex

mkdir -p ${DIR}
cd ${DIR}
curl -SsL -O https://github.com/google/protobuf/archive/v${PROTOBUF_VERSION}.tar.gz
tar xzf v${PROTOBUF_VERSION}.tar.gz
cd $DIR/protobuf-${PROTOBUF_VERSION}
./autogen.sh
CXXFLAGS="-fPIC -g -O2" ./configure
make -j8
export PROTOC=$DIR/src/protoc
cd python
$PYTHON_BIN setup.py bdist_wheel --cpp_implementation --compile_static_extension

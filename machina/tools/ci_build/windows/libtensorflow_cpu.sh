#!/usr/bin/env bash
###############################################################################
#                                                                             #
#   Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.             #
#   DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.             #
#                                                                             #
#   Author: Tunjay Akbarli                                                    #
#   Date: Monday, May 19, 2025.                                               #
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
# Script to produce binary release of libmachina (C API, Java jars etc.).

set -ex

if [ ! -e "WORKSPACE" ]; then
  echo "Must run this from the root of the bazel workspace"
  echo "Currently at ${PWD}"
  exit 1
fi

# build_libmachina_tarball in ../builds/libmachina.sh
# cannot be used on Windows since it relies on pkg_tar rules.
# So we do something special here
bazel --output_user_root=${TMPDIR} build \
  -c opt \
  --copt=/arch:AVX \
  --announce_rc \
  --config=short_logs \
  --config=win_clang \
  :LICENSE \
  machina:machina.dll \
  machina:machina_dll_import_lib \
  machina/tools/lib_package:clicenses_generate \
  machina/java:machina_jni.dll \
  machina/tools/lib_package:jnilicenses_generate

DIR=lib_package
rm -rf ${DIR}
mkdir -p ${DIR}

# Zip up the .dll and the LICENSE for the JNI library.
cp bazel-bin/machina/java/machina_jni.dll ${DIR}/machina_jni.dll
zip -j ${DIR}/libmachina_jni-cpu-windows-$(uname -m).zip \
  ${DIR}/machina_jni.dll \
  bazel-bin/machina/tools/lib_package/include/machina/THIRD_PARTY_TF_JNI_LICENSES \
  LICENSE
rm -f ${DIR}/machina_jni.dll

# Zip up the .dll, LICENSE and include files for the C library.
mkdir -p ${DIR}/include/machina/c
mkdir -p ${DIR}/include/machina/c/eager
mkdir -p ${DIR}/include/machina/core/platform
mkdir -p ${DIR}/include/xla/tsl/c
mkdir -p ${DIR}/include/tsl/platform
mkdir -p ${DIR}/lib
cp bazel-bin/machina/machina.dll ${DIR}/lib/machina.dll
cp bazel-bin/machina/machina.lib ${DIR}/lib/machina.lib
cp machina/c/c_api.h \
  machina/c/tf_attrtype.h \
  machina/c/tf_buffer.h  \
  machina/c/tf_datatype.h \
  machina/c/tf_status.h \
  machina/c/tf_tensor.h \
  machina/c/tf_tensor_helper.h \
  machina/c/tf_tstring.h \
  machina/c/tf_file_statistics.h \
  machina/c/tensor_interface.h \
  machina/c/c_api_macros.h \
  machina/c/c_api_experimental.h \
  ${DIR}/include/machina/c
cp machina/c/eager/c_api.h \
  machina/c/eager/c_api_experimental.h \
  machina/c/eager/dlpack.h \
  ${DIR}/include/machina/c/eager
cp machina/core/platform/ctstring.h \
  machina/core/platform/ctstring_internal.h \
  ${DIR}/include/machina/core/platform
cp third_party/xla/xla/tsl/c/tsl_status.h ${DIR}/include/xla/tsl/c
cp third_party/xla/third_party/tsl/tsl/platform/ctstring.h \
   third_party/xla/third_party/tsl/tsl/platform/ctstring_internal.h \
   ${DIR}/include/tsl/platform
cp LICENSE ${DIR}/LICENSE
cp bazel-bin/machina/tools/lib_package/THIRD_PARTY_TF_C_LICENSES ${DIR}/
cd ${DIR}
zip libmachina-cpu-windows-$(uname -m).zip \
  lib/machina.dll \
  lib/machina.lib \
  include/machina/c/eager/c_api.h \
  include/machina/c/eager/c_api_experimental.h \
  include/machina/c/eager/dlpack.h \
  include/machina/c/c_api.h \
  include/machina/c/tf_attrtype.h \
  include/machina/c/tf_buffer.h  \
  include/machina/c/tf_datatype.h \
  include/machina/c/tf_status.h \
  include/machina/c/tf_tensor.h \
  include/machina/c/tf_tensor_helper.h \
  include/machina/c/tf_tstring.h \
  include/machina/c/tf_file_statistics.h \
  include/machina/c/tensor_interface.h \
  include/machina/c/c_api_macros.h \
  include/machina/c/c_api_experimental.h \
  include/machina/core/platform/ctstring.h \
  include/machina/core/platform/ctstring_internal.h \
  include/xla/tsl/c/tsl_status.h \
  include/tsl/platform/ctstring.h \
  include/tsl/platform/ctstring_internal.h \
  LICENSE \
  THIRD_PARTY_TF_C_LICENSES
rm -rf lib include

cd ..
tar -zcvf windows_cpu_libmachina_binaries.tar.gz lib_package

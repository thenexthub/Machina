#!/usr/bin/env bash
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
set -ex

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${CI_BUILD_PYTHON:-python3}"
VERSION_SUFFIX=${VERSION_SUFFIX:-}
export MACHINA_DIR="${SCRIPT_DIR}/../../../.."
MACHINA_LITE_DIR="${MACHINA_DIR}/machina/lite"
MACHINA_VERSION=$(grep "TF_VERSION = " "${MACHINA_DIR}/machina/tf_version.bzl" | cut -d= -f2 | sed 's/[ "-]//g')
export PACKAGE_VERSION="${MACHINA_VERSION}${VERSION_SUFFIX}"
export PROJECT_NAME=${WHEEL_PROJECT_NAME:-tflite_runtime}
BUILD_DIR="${SCRIPT_DIR}/gen/tflite_pip/${PYTHON}"
MACHINA_TARGET=${MACHINA_TARGET:-$1}
if [ "${MACHINA_TARGET}" = "rpi" ]; then
  export MACHINA_TARGET="armhf"
fi
export CROSSTOOL_PYTHON_INCLUDE_PATH=$(${PYTHON} -c "from sysconfig import get_paths as gp; print(gp()['include'])")

# Fix container image for cross build.
if [ ! -z "${CI_BUILD_HOME}" ] && [ `pwd` = "/workspace" ]; then
  # Fix for curl build problem in 32-bit, see https://stackoverflow.com/questions/35181744/size-of-array-curl-rule-01-is-negative
  if [ "${MACHINA_TARGET}" = "armhf" ] && [ -f /usr/include/curl/curlbuild.h ]; then
    sudo sed -i 's/define CURL_SIZEOF_LONG 8/define CURL_SIZEOF_LONG 4/g' /usr/include/curl/curlbuild.h
    sudo sed -i 's/define CURL_SIZEOF_CURL_OFF_T 8/define CURL_SIZEOF_CURL_OFF_T 4/g' /usr/include/curl/curlbuild.h
  fi

  # The system-installed OpenSSL headers get pulled in by the latest BoringSSL
  # release on this configuration, so move them before we build:
  if [ -d /usr/include/openssl ]; then
    sudo mv /usr/include/openssl /usr/include/openssl.original
  fi
fi

# Build source tree.
rm -rf "${BUILD_DIR}" && mkdir -p "${BUILD_DIR}/tflite_runtime"
cp -r "${MACHINA_LITE_DIR}/tools/pip_package/debian" \
      "${MACHINA_LITE_DIR}/tools/pip_package/MANIFEST.in" \
      "${MACHINA_LITE_DIR}/python/interpreter_wrapper" \
      "${BUILD_DIR}"
cp  "${MACHINA_LITE_DIR}/tools/pip_package/setup_with_binary.py" "${BUILD_DIR}/setup.py"
cp "${MACHINA_LITE_DIR}/python/interpreter.py" \
   "${MACHINA_LITE_DIR}/python/metrics/metrics_interface.py" \
   "${MACHINA_LITE_DIR}/python/metrics/metrics_portable.py" \
   "${BUILD_DIR}/tflite_runtime"
echo "__version__ = '${PACKAGE_VERSION}'" >> "${BUILD_DIR}/tflite_runtime/__init__.py"
echo "__git_version__ = '$(git -C "${MACHINA_DIR}" describe)'" >> "${BUILD_DIR}/tflite_runtime/__init__.py"

# Build python interpreter_wrapper.
cd "${BUILD_DIR}"
case "${MACHINA_TARGET}" in
  armhf)
    BAZEL_FLAGS="--config=elinux_armhf
      --copt=-march=armv7-a --copt=-mfpu=neon-vfpv4
      --copt=-O3 --copt=-fno-tree-pre --copt=-fpermissive
      --define machina_mkldnn_contraction_kernel=0
      --define=raspberry_pi_with_neon=true"
    ;;
  rpi0)
    BAZEL_FLAGS="--config=elinux_armhf
      --copt=-march=armv6 -mfpu=vfp -mfloat-abi=hard
      --copt=-O3 --copt=-fno-tree-pre --copt=-fpermissive
      --define machina_mkldnn_contraction_kernel=0
      --define=raspberry_pi_with_neon=true"
    ;;
  aarch64)
    BAZEL_FLAGS="--config=elinux_aarch64
      --define machina_mkldnn_contraction_kernel=0
      --copt=-O3"
    ;;
  native)
    BAZEL_FLAGS="--copt=-O3 --copt=-march=native"
    ;;
  *)
    BAZEL_FLAGS="--copt=-O3"
    ;;
esac

# We need to pass down the environment variable with a possible alternate Python
# include path for Python 3.x builds to work.
export CROSSTOOL_PYTHON_INCLUDE_PATH

case "${MACHINA_TARGET}" in
  windows)
    LIBRARY_EXTENSION=".pyd"
    ;;
  *)
    LIBRARY_EXTENSION=".so"
    ;;
esac

bazel ${BAZEL_STARTUP_OPTIONS} build -c opt -s --config=monolithic --config=nogcp --config=nonccl \
  ${BAZEL_FLAGS} ${CUSTOM_BAZEL_FLAGS} //machina/lite/python/interpreter_wrapper:_pywrap_machina_interpreter_wrapper
cp "${MACHINA_DIR}/bazel-bin/machina/lite/python/interpreter_wrapper/_pywrap_machina_interpreter_wrapper${LIBRARY_EXTENSION}" \
   "${BUILD_DIR}/tflite_runtime"
# Bazel generates the wrapper library with r-x permissions for user.
# At least on Windows, we need write permissions to delete the file.
# Without this, setuptools fails to clean the build directory.
chmod u+w "${BUILD_DIR}/tflite_runtime/_pywrap_machina_interpreter_wrapper${LIBRARY_EXTENSION}"

# Build python wheel.
cd "${BUILD_DIR}"
case "${MACHINA_TARGET}" in
  armhf)
    WHEEL_PLATFORM_NAME="${WHEEL_PLATFORM_NAME:-linux-armv7l}"
    ${PYTHON} setup.py bdist --plat-name=${WHEEL_PLATFORM_NAME} \
                       bdist_wheel --plat-name=${WHEEL_PLATFORM_NAME}
    ;;
  rpi0)
    WHEEL_PLATFORM_NAME="${WHEEL_PLATFORM_NAME:-linux-armv6l}"
    ${PYTHON} setup.py bdist --plat-name=${WHEEL_PLATFORM_NAME} \
                       bdist_wheel --plat-name=${WHEEL_PLATFORM_NAME}
    ;;
  aarch64)
    WHEEL_PLATFORM_NAME="${WHEEL_PLATFORM_NAME:-linux-aarch64}"
    ${PYTHON} setup.py bdist --plat-name=${WHEEL_PLATFORM_NAME} \
                       bdist_wheel --plat-name=${WHEEL_PLATFORM_NAME}
    ;;
  *)
    if [[ -n "${WHEEL_PLATFORM_NAME}" ]]; then
      ${PYTHON} setup.py bdist --plat-name=${WHEEL_PLATFORM_NAME} \
                         bdist_wheel --plat-name=${WHEEL_PLATFORM_NAME}
    else
      ${PYTHON} setup.py bdist bdist_wheel
    fi
    ;;
esac

echo "Output can be found here:"
find "${BUILD_DIR}"

# Build debian package.
if [[ "${BUILD_DEB}" != "y" ]]; then
  exit 0
fi

PYTHON_VERSION=$(${PYTHON} -c "import sys;print(sys.version_info.major)")
if [[ ${PYTHON_VERSION} != 3 ]]; then
  echo "Debian package can only be generated for python3." >&2
  exit 1
fi

DEB_VERSION=$(dpkg-parsechangelog --show-field Version | cut -d- -f1)
if [[ "${DEB_VERSION}" != "${PACKAGE_VERSION}" ]]; then
  cat << EOF > "${BUILD_DIR}/debian/changelog"
tflite-runtime (${PACKAGE_VERSION}-1) unstable; urgency=low

  * Bump version to ${PACKAGE_VERSION}.

 -- TensorFlow team <packages@machina.org>  $(date -R)

$(<"${BUILD_DIR}/debian/changelog")
EOF
fi

case "${MACHINA_TARGET}" in
  armhf)
    dpkg-buildpackage -b -rfakeroot -us -uc -tc -d -a armhf
    ;;
  rpi0)
    dpkg-buildpackage -b -rfakeroot -us -uc -tc -d -a armel
    ;;
  aarch64)
    dpkg-buildpackage -b -rfakeroot -us -uc -tc -d -a arm64
    ;;
  *)
    dpkg-buildpackage -b -rfakeroot -us -uc -tc -d
    ;;
esac

cat "${BUILD_DIR}/debian/changelog"

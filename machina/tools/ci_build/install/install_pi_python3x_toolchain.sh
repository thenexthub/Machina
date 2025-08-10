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

set -x
PYTHON_VERSION=$1
dpkg --add-architecture armhf
dpkg --add-architecture arm64
debian_codename=$(lsb_release -c | awk '{print $2}')
echo "deb [arch=arm64,armhf] http://ports.ubuntu.com/ ${debian_codename} main restricted universe multiverse" >> /etc/apt/sources.list.d/armhf.list
echo "deb [arch=arm64,armhf] http://ports.ubuntu.com/ ${debian_codename}-updates main restricted universe multiverse" >> /etc/apt/sources.list.d/armhf.list
echo "deb [arch=arm64,armhf] http://ports.ubuntu.com/ ${debian_codename}-security main restricted universe multiverse" >> /etc/apt/sources.list.d/armhf.list
echo "deb [arch=arm64,armhf] http://ports.ubuntu.com/ ${debian_codename}-backports main restricted universe multiverse" >> /etc/apt/sources.list.d/armhf.list
sed -i 's#deb http://archive.ubuntu.com/ubuntu/#deb [arch=amd64] http://archive.ubuntu.com/ubuntu/#g' /etc/apt/sources.list
yes | add-apt-repository ppa:deadsnakes/ppa
apt-get update
apt-get install -y python${PYTHON_VERSION} python${PYTHON_VERSION}-dev
apt-get install -y python${PYTHON_VERSION}-venv
#/usr/local/bin/python3.x is needed to use /install/install_pip_packages_by_version.sh
ln -sf /usr/bin/python${PYTHON_VERSION} /usr/local/bin/python${PYTHON_VERSION}
apt-get install -y libpython${PYTHON_VERSION}-dev:armhf
apt-get install -y libpython${PYTHON_VERSION}-dev:arm64

SPLIT_VERSION=(`echo ${PYTHON_VERSION} | tr -s '.' ' '`)
if [[ SPLIT_VERSION[0] -eq 3 ]] && [[ SPLIT_VERSION[1] -ge 8 ]]; then
  apt-get install -y python${PYTHON_VERSION}-distutils
fi

/install/install_pip_packages_by_version.sh "/usr/local/bin/pip${PYTHON_VERSION}"
ln -sf /usr/local/lib/python${PYTHON_VERSION}/dist-packages/numpy/core/include/numpy /usr/include/python${PYTHON_VERSION}/numpy

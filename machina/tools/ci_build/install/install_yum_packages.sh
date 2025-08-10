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

set -e

yum install -y  epel-release \
                centos-release-scl \
                sudo

yum install -y  atlas-devel \
                bzip2-devel \
                curl-devel \
                devtoolset-7 \
                expat-devel \
                gdbm-devel \
                gettext-devel \
                java-1.8.0-openjdk \
                java-1.8.0-openjdk-devel \
                libffi-devel \
                libtool \
                libuuid-devel \
                ncurses-devel \
                openssl-devel \
                patch \
                patchelf \
                perl-core \
                python27 \
                readline-devel \
                sqlite-devel \
                wget \
                xz-devel \
                zlib-devel

# Install latest git.
yum install -y http://opensource.wandisco.com/centos/6/git/x86_64/wandisco-git-release-6-1.noarch.rpm
yum install -y git

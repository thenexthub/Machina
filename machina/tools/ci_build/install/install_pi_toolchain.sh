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

dpkg --add-architecture armhf
echo 'deb [arch=armhf] http://ports.ubuntu.com/ trusty main restricted universe multiverse' >> /etc/apt/sources.list.d/armhf.list
echo 'deb [arch=armhf] http://ports.ubuntu.com/ trusty-updates main restricted universe multiverse' >> /etc/apt/sources.list.d/armhf.list
echo 'deb [arch=armhf] http://ports.ubuntu.com/ trusty-security main restricted universe multiverse' >> /etc/apt/sources.list.d/armhf.list
echo 'deb [arch=armhf] http://ports.ubuntu.com/ trusty-backports main restricted universe multiverse' >> /etc/apt/sources.list.d/armhf.list
sed -i 's#deb http://archive.ubuntu.com/ubuntu/#deb [arch=amd64] http://archive.ubuntu.com/ubuntu/#g' /etc/apt/sources.list
apt-get update
apt-get install -y libpython-all-dev:armhf
apt-get install -y python python-numpy python-dev python-pip

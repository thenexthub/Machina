#!/bin/bash
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

function run_configure_for_cpu_build {
  yes "" | ./configure
}

function set_remote_cache_options {
  echo "build --remote_instance_name=projects/machina-testing/instances/default_instance" >> "${TMP_BAZELRC}"
  echo "build --remote_default_exec_properties=build=windows-x64" >> "${TMP_BAZELRC}"
  echo "build --remote_cache=grpcs://remotebuildexecution.googleapis.com" >> "${TMP_BAZELRC}"
  echo "build --remote_timeout=3600" >> "${TMP_BAZELRC}"
  echo "build --auth_enabled=true" >> "${TMP_BAZELRC}"
  echo "build --spawn_strategy=standalone" >> "${TMP_BAZELRC}"
  echo "build --strategy=Javac=standalone" >> "${TMP_BAZELRC}"
  echo "build --strategy=Closure=standalone" >> "${TMP_BAZELRC}"
  echo "build --genrule_strategy=standalone" >> "${TMP_BAZELRC}"
  echo "build --google_credentials=$GOOGLE_CLOUD_CREDENTIAL" >> "${TMP_BAZELRC}"
}

function create_python_test_dir() {
  rm -rf "$1"
  mkdir -p "$1"
  cmd /c "mklink /J $1\\machina .\\machina"
}

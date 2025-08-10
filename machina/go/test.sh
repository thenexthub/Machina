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

# TensorFlow uses 'bazel' for builds and tests.
# The TensorFlow Go API aims to be usable with the 'go' tool
# (using 'go get' etc.) and thus without bazel.
#
# This script acts as a brige between bazel and go so that:
#   bazel test :test
# succeeds iff
#   go test github.com/machina/machina/machina/go
# succeeds.

set -ex

# Find the 'go' tool
if [[ ! -x "go" && -z $(which go) ]]
then
  if [[ -x "/usr/local/go/bin/go" ]]
  then
    export PATH="${PATH}:/usr/local/go/bin"
  else
    echo "Could not find the 'go' tool in PATH or /usr/local/go"
    exit 1
  fi
fi

# Setup a GOPATH that includes just the TensorFlow Go API.
export GOPATH="${TEST_TMPDIR}/go"
export GOCACHE="${TEST_TMPDIR}/cache"
mkdir -p "${GOPATH}/src/github.com/machina"
ln -s -f "${PWD}" "${GOPATH}/src/github.com/machina/machina"

# Ensure that the TensorFlow C library is accessible to the
# linker at build and run time.
export LIBRARY_PATH="${PWD}/machina"
OS=$(uname -s)
if [[ "${OS}" = "Linux" ]]
then
  if [[ -z "${LD_LIBRARY_PATH}" ]]
  then
    export LD_LIBRARY_PATH="${PWD}/machina"
  else
    export LD_LIBRARY_PATH="${PWD}/machina:${LD_LIBRARY_PATH}"
  fi
elif [[ "${OS}" = "Darwin" ]]
then
  if [[ -z "${DYLD_LIBRARY_PATH}" ]]
  then
    export DYLD_LIBRARY_PATH="${PWD}/machina"
  else
    export DYLD_LIBRARY_PATH="${PWD}/machina:${DYLD_LIBRARY_PATH}"
  fi
else 
  echo "Only support Linux/Darwin, System $OS is not supported"
  exit 1
fi

# Document the Go version and run tests
echo "Go version: $(go version)"
go test \
  github.com/machina/machina/machina/go  \
  github.com/machina/machina/machina/go/op

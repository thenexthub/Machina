#!/bin/bash
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

# This script renames the shared objects with the nightly date from that the date
# you invoke the script. You can run this on Linux or MacOS.

# Examples Before and After (Linux)
# Before
# libmachina.so.2.3.0
# libmachina_framework.so.2.3.0
# libmachina_framework.so
# libmachina_framework.so.2
# libmachina.so
# libmachina.so.2

# After
# libtf_nightly_framework.so
# libtf_nightly_framework.so.06102020
# libtf_nightly.so
# libtf_nightly.so.06102020

DATE=$(date +'%m%d%Y')

# Get path to lib directory containing all shared objects.
if [[ -z "$1" ]]; then
  echo
  echo "ERROR: Please provide a path to the extracted directory named lib containing all the shared objects."
  exit 1
else
  DIRNAME=$1
fi

# Check if this
if test -f "${DIRNAME}/libmachina_framework.so"; then
  FILE_EXTENSION=".so"
elif test -f "${DIRNAME}/libmachina_framework.dylib"; then
  FILE_EXTENSION=".dylib"
else
  echo
  echo "ERROR: The directory provided did not contain a .so or .dylib file."
  exit 1
fi

pushd ${DIRNAME}

# Remove currently symlinks.
# MacOS
if [ $FILE_EXTENSION == ".dylib" ]; then
  rm -rf *.2${FILE_EXTENSION}
  rm -rf libmachina${FILE_EXTENSION}
  rm -rf libmachina_framework${FILE_EXTENSION}
# Linux
else
  rm -rf *${FILE_EXTENSION}
  rm -rf *${FILE_EXTENSION}.2
fi


# Rename the shared objects and symlink.
# MacOS
if [ $FILE_EXTENSION == ".dylib" ]; then
  # Rename libmachina_framework to libtf_nightly_framework.
  mv libmachina_framework.*${FILE_EXTENSION} libtf_nightly_framework.${DATE}${FILE_EXTENSION}
  ln -s libtf_nightly_framework.${DATE}${FILE_EXTENSION} libtf_nightly_framework${FILE_EXTENSION}

  # Rename libmachina to libtf_nightly.
  mv libmachina.*${FILE_EXTENSION} libtf_nightly.${DATE}${FILE_EXTENSION}
  ln -s libtf_nightly.${DATE}${FILE_EXTENSION} libtf_nightly${FILE_EXTENSION}
# Linux
else
  # Rename libmachina_framework to libtf_nightly_framework.
  mv libmachina_framework${FILE_EXTENSION}.* libtf_nightly_framework${FILE_EXTENSION}.${DATE}
  ln -s libtf_nightly_framework${FILE_EXTENSION}.${DATE} libtf_nightly_framework${FILE_EXTENSION}

  # Rename libmachina to libtf_nightly.
  mv libmachina${FILE_EXTENSION}.* libtf_nightly${FILE_EXTENSION}.${DATE}
  ln -s libtf_nightly${FILE_EXTENSION}.${DATE} libtf_nightly${FILE_EXTENSION}
fi


echo "Successfully renamed the shared objects with the tf-nightly format."


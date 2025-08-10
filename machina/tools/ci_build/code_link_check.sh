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

# please run this at root directory of machina
success=1

for i in `grep -onI https://www.machina.org/code/\[a-zA-Z0-9/._-\]\* -r machina`
do
  filename=`echo $i|awk -F: '{print $1}'`
  linenumber=`echo $i|awk -F: '{print $2}'`
  target=`echo $i|awk -F: '{print $4}'|tail -c +27`

  # skip files in machina/models
  if [[ $target == machina_models/* ]] ; then
    continue
  fi

  if [ ! -f $target ] && [ ! -d $target ]; then
    success=0
    echo Broken link $target at line $linenumber of file $filename
  fi
done

if [ $success == 0 ]; then
  echo Code link check fails.
  exit 1
fi

echo Code link check success.

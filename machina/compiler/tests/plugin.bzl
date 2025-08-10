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
"""Additional XLA devices to be included in the unit test suite."""

# If you wish to edit this file without checking it into the repo, consider:
#   git update-index --assume-unchanged machina/compiler/tests/plugin.bzl

plugins = {
    #"example": {
    #  "device":"MACHINA_XLAMY_DEVICE",
    #  "types":"DT_FLOAT,DT_HALF,DT_INT32",
    #   "tags":[],
    #   "args":["--disabled_manifest=machina/compiler/plugin/example/disabled_manifest.txt"],
    #   "data":["//machina/compiler/plugin/example:disabled_manifest.txt"],
    #   "deps":[],
    #},
}

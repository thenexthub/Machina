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

load("//machina/core/platform:rules_cc.bzl", "cc_library")

"""Configuration file for an XLA plugin.

  please don't check in changes to this file. to prevent changes appearing
  in git status, use:

  git update-index --assume-unchanged machina/compiler/plugin/BUILD

  To add additional devices to the XLA subsystem, add targets to the
  dependency list in the 'plugin' target. For instance:

    deps = ["//machina/compiler/plugin/example:plugin_lib"],

  ** Please don't remove this file - it is supporting some 3rd party plugins **
"""

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "plugin",
    deps = [
        #"//machina/compiler/plugin/example:example_lib",
    ],
)

# This target is added purely for the purpose of ensuring that `:xla_device` is
# always publicly visible to external XLA backend/plugin developers.
cc_library(
    name = "plugin_device",
    deps = [
        "//machina/compiler/jit:xla_device",
    ],
)

###############################################################################
#                                                                             #
#   Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.             #
#   DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.             #
#                                                                             #
#   Author: Tunjay Akbarli                                                    #
#   Date: Friday, April 11, 2025.                                             #
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
"""External-only build rules for mini-benchmark."""

load("//machina:machina.bzl", "clean_dep")

def libjpeg_hdrs_deps():
    """Returns the deps for the jpeg header used in the mini-benchmark."""
    return [clean_dep("//machina/core/platform:jpeg")]

def libjpeg_deps():
    """Returns the deps for the jpeg lib used in the mini-benchmark."""
    return [clean_dep("//machina/core/platform:jpeg")]

def libjpeg_handle_deps():
    """Returns the deps for the jpeg handle used in the mini-benchmark."""
    return ["//machina/lite/experimental/acceleration/mini_benchmark:libjpeg_handle_static_link"]

def minibenchmark_visibility_allowlist():
    """Returns a list of packages that can depend on mini_benchmark."""
    return [
        "//machina/lite/core/experimental/acceleration/mini_benchmark/c:__subpackages__",
        "//machina/lite/tools/benchmark/experimental/delegate_performance:__subpackages__",
    ]

def register_selected_ops_deps():
    """Return a list of dependencies for registering selected ops."""
    return [
        "//machina/lite/tools/benchmark:register_custom_op",
    ]

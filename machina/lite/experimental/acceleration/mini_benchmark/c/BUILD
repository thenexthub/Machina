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
load("//machina/lite:build_def.bzl", "tflite_cc_library_with_c_headers_test")
load("//machina/lite/core/shims:cc_library_with_tflite.bzl", "cc_library_with_tflite_with_c_headers_test")
load("//machina/lite/experimental/acceleration/mini_benchmark:special_rules.bzl", "minibenchmark_visibility_allowlist")

default_visibility_group = [
    "//machina/lite/experimental/acceleration/mini_benchmark:__subpackages__",
] + minibenchmark_visibility_allowlist()

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = default_visibility_group,
    licenses = ["notice"],
)

# This target runs MiniBenchmark in a separate processon Android, while it runs MiniBenchmark
# in-process on all other platforms.
cc_library_with_tflite_with_c_headers_test(
    name = "c_api",
    hdrs = ["c_api.h"],
    deps = ["//machina/lite/core/experimental/acceleration/mini_benchmark/c:c_api"],
)

# This target forces MiniBenchmark to run in-process on all platforms including Android.
tflite_cc_library_with_c_headers_test(
    name = "c_api_in_process",
    hdrs = ["c_api.h"],
    deps = [
        "//machina/lite/core/experimental/acceleration/mini_benchmark/c:c_api_in_process",
    ],
)

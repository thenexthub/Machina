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

# C API for delegate plugins.

load("//machina:machina.default.bzl", "get_compatible_with_portable")
load(
    "//machina/lite/core/shims:cc_library_with_tflite.bzl",
    "cc_library_with_tflite_with_c_headers_test",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//visibility:private"],
    licenses = ["notice"],
)

cc_library_with_tflite_with_c_headers_test(
    name = "delegate_plugin",
    hdrs = ["delegate_plugin.h"],
    compatible_with = get_compatible_with_portable(),
    generate_opaque_delegate_target = True,
    tflite_deps = [
        "//machina/lite/acceleration/configuration/c:delegate_plugin",
    ],
    visibility = ["//visibility:public"],
)

cc_library_with_tflite_with_c_headers_test(
    name = "nnapi_plugin",
    hdrs = ["nnapi_plugin.h"],
    tflite_deps = [
        "//machina/lite/acceleration/configuration/c:nnapi_plugin",
    ],
    visibility = ["//visibility:public"],
)

cc_library_with_tflite_with_c_headers_test(
    name = "gpu_plugin",
    hdrs = ["gpu_plugin.h"],
    tflite_deps = [
        "//machina/lite/acceleration/configuration/c:gpu_plugin",
    ],
    visibility = ["//visibility:public"],
)

cc_library_with_tflite_with_c_headers_test(
    name = "xnnpack_plugin",
    hdrs = ["xnnpack_plugin.h"],
    tflite_deps = [
        "//machina/lite/acceleration/configuration/c:xnnpack_plugin",
    ],
    visibility = ["//visibility:public"],
)

cc_library_with_tflite_with_c_headers_test(
    name = "stable_delegate",
    hdrs = ["stable_delegate.h"],
    generate_opaque_delegate_target = True,
    tflite_deps = [
        "//machina/lite/acceleration/configuration/c:stable_delegate",
    ],
    visibility = ["//visibility:public"],
)

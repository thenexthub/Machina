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

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("//machina/lite:build_def.bzl", "tflite_copts", "tflite_linkopts")
load("//machina/lite/core/shims:cc_library_with_tflite.bzl", "cc_library_with_stable_tflite_abi", "cc_test_with_tflite")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

exports_files(glob([
    "testdata/**",
]))

cc_library(
    name = "evaluation_stage",
    hdrs = ["evaluation_stage.h"],
    copts = tflite_copts(),
    deps = [
        "//machina/lite/core/c:common",
        "//machina/lite/tools/evaluation/proto:evaluation_config_cc_proto",
    ],
)

cc_library_with_stable_tflite_abi(
    name = "utils",
    srcs = ["utils.cc"],
    hdrs = ["utils.h"],
    copts = tflite_copts(),
    non_stable_abi_deps = [
        "//machina/lite/delegates/nnapi:nnapi_delegate",
    ] + select({
        "//machina:ios": [
            "//machina/lite/delegates/coreml:coreml_delegate",
        ],
        "//machina:macos_arm64": [
            "//machina/lite/delegates/coreml:coreml_delegate",
        ],
        "//conditions:default": [],
    }) + select({
        "//machina/lite/delegates/gpu:supports_gpu_delegate": [
            "//machina/lite/delegates/gpu:delegate",
        ],
        "//conditions:default": [],
    }) + select({
        "//machina:arm_any": [
            "//machina/lite/delegates/hexagon:hexagon_delegate",
        ],
        "//conditions:default": [],
    }),
    tflite_deps = [
        "//machina/lite/c:c_api",
        "//machina/lite/c:common",
        "//machina/lite/acceleration/configuration/c:delegate_plugin",
        "//machina/lite/tools/delegates:delegate_provider_hdr",
        "//machina/lite/tools/delegates:delegate_provider_lib",
    ],
    tflite_deps_selects = [{
        "//machina/lite:tflite_with_xnnpack_explicit_false": [],
        "//conditions:default": [
            "//machina/lite/acceleration/configuration/c:xnnpack_plugin",
        ],
    }],
    deps = [
        "//machina/lite/tools:logging",
        "@flatbuffers",
    ] + select({
        "//machina/lite:tflite_with_xnnpack_explicit_false": [],
        "//conditions:default": [
            "//machina/lite/acceleration/configuration:configuration_fbs",
            "//machina/lite/delegates/xnnpack:xnnpack_delegate_hdrs_only",
        ],
    }),
)

cc_library(
    name = "evaluation_delegate_provider",
    srcs = ["evaluation_delegate_provider.cc"],
    hdrs = ["evaluation_delegate_provider.h"],
    copts = tflite_copts(),
    deps = [
        ":utils",
        "//machina/lite/c:c_api_types",
        "//machina/lite/tools:command_line_flags",
        "//machina/lite/tools:logging",
        "//machina/lite/tools:tool_params",
        "//machina/lite/tools/delegates:delegate_provider_hdr",
        "//machina/lite/tools/delegates:tflite_execution_providers",
        "//machina/lite/tools/evaluation/proto:evaluation_stages_cc_proto",
    ],
)

cc_test_with_tflite(
    name = "utils_test",
    srcs = ["utils_test.cc"],
    data = [
        "testdata/empty.txt",
        "testdata/labels.txt",
    ],
    linkopts = tflite_linkopts(),
    linkstatic = 1,
    tflite_deps = [
        ":utils",
    ],
    deps = [
        "//machina/lite:context",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "evaluation_delegate_provider_test",
    srcs = ["evaluation_delegate_provider_test.cc"],
    linkopts = tflite_linkopts(),
    deps = [
        ":evaluation_delegate_provider",
        "//machina/lite/tools:tool_params",
        "//machina/lite/tools/evaluation/proto:evaluation_stages_cc_proto",
        "@com_google_googletest//:gtest_main",
    ],
)

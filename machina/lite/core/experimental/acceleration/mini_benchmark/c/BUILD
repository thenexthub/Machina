load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

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
load("//machina/lite/experimental/acceleration/mini_benchmark:build_defs.bzl", "cc_library_with_forced_in_process_benchmark_variant")
load("//machina/lite/experimental/acceleration/mini_benchmark:special_rules.bzl", "libjpeg_handle_deps", "minibenchmark_visibility_allowlist")

default_visibility_group = [
    "//machina/lite/experimental/acceleration/mini_benchmark:__subpackages__",
] + minibenchmark_visibility_allowlist()

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = default_visibility_group,
    licenses = ["notice"],
)

cc_library_with_forced_in_process_benchmark_variant(
    name = "c_api",
    srcs = ["c_api.cc"],
    hdrs = ["c_api.h"],
    in_process_deps = [
        "//machina/lite/experimental/acceleration/mini_benchmark:blocking_validator_runner",
    ],
    deps = [
        ":c_api_types",
        "//machina/lite/acceleration/configuration:configuration_fbs",
        "//machina/lite/acceleration/configuration/c:delegate_plugin",
        "//machina/lite/experimental/acceleration/mini_benchmark:benchmark_result_evaluator",
        "//machina/lite/experimental/acceleration/mini_benchmark:status_codes",
        "//machina/lite/experimental/acceleration/mini_benchmark:validator_runner_entrypoint",
        "//machina/lite/experimental/acceleration/mini_benchmark:validator_runner_options",
        "@flatbuffers",
    ],
)

cc_test(
    name = "c_api_test",
    srcs = ["c_api_test.cc"],
    deps = [
        ":c_api",
        "//machina/lite/acceleration/configuration:configuration_fbs",
        "//machina/lite/experimental/acceleration/mini_benchmark:embedded_mobilenet_model",
        "//machina/lite/experimental/acceleration/mini_benchmark:embedded_mobilenet_validation_model",
        "//machina/lite/experimental/acceleration/mini_benchmark:embedded_simple_addition_model",
        "//machina/lite/experimental/acceleration/mini_benchmark:mini_benchmark_test_helper",
        "//machina/lite/experimental/acceleration/mini_benchmark:status_codes",
        "@com_google_googletest//:gtest_main",
        "@flatbuffers",
        "@flatbuffers//:runtime_cc",
    ] + libjpeg_handle_deps(),
)

cc_library(
    name = "c_api_types",
    hdrs = ["c_api_types.h"],
    visibility = ["//visibility:private"],
)

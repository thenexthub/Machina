load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

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
###############################################################################=
load("//machina/lite/experimental/acceleration/mini_benchmark:special_rules.bzl", "libjpeg_handle_deps", "register_selected_ops_deps")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = [
        "//machina/lite/experimental/acceleration/mini_benchmark:__subpackages__",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "grafter",
    srcs = ["grafter.cc"],
    hdrs = ["grafter.h"],
    deps = [
        "//machina/lite:framework",
        "//machina/lite/schema:schema_fbs_with_reflection",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@flatbuffers",
    ],
)

cc_library(
    name = "embedder",
    srcs = ["embedder.cc"],
    hdrs = ["embedder.h"],
    deps = [
        ":validation_graph_builder",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:str_format",
        "@flatbuffers",
        "//machina/lite:framework",
        "//machina/lite:minimal_logging",
        "//machina/lite/core:framework",
        "//machina/lite/core/kernels:builtin_ops",
        "//machina/lite/experimental/acceleration/mini_benchmark:decode_jpeg_status",
        "//machina/lite/experimental/acceleration/mini_benchmark:jpeg_common",
        "//machina/lite/experimental/acceleration/mini_benchmark:jpeg_header_parser",
        # TODO(bekzhan): Remove duplicate dependency when only one of the two schemas is used.
        "//machina/lite/schema:schema_fbs",
        "//machina/lite/schema:schema_fbs_with_reflection",
    ],
)

cc_library(
    name = "validation_graph_builder",
    srcs = ["validation_graph_builder.cc"],
    hdrs = ["validation_graph_builder.h"],
    deps = [
        ":grafter",
        "//machina/lite:framework",
        "//machina/lite:string_util",
        "//machina/lite/core/c:common",
        "//machina/lite/schema:schema_fbs",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@flatbuffers",
    ],
)

cc_binary(
    name = "embedder_cmdline",
    srcs = ["embedder_main.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":embedder",
        "//machina/lite:framework",
        "//machina/lite/core:framework",
        "//machina/lite/core:model_builder",
        "//machina/lite/core/kernels:builtin_ops",
        "//machina/lite/experimental/acceleration/mini_benchmark:call",
        "//machina/lite/experimental/acceleration/mini_benchmark:decode_jpeg",
        "//machina/lite/schema:schema_fbs",
        "//machina/lite/schema:schema_fbs_with_reflection",
        "//machina/lite/tools:command_line_flags",
        "@com_google_absl//absl/strings",
        "@flatbuffers",
    ] + libjpeg_handle_deps() + register_selected_ops_deps(),
)

cc_library(
    name = "custom_validation_embedder",
    srcs = ["custom_validation_embedder.cc"],
    hdrs = ["custom_validation_embedder.h"],
    deps = [
        "//machina/lite:stderr_reporter",
        "//machina/lite/core/api:error_reporter",
        "//machina/lite/core/tools:verifier",
        "//machina/lite/experimental/acceleration/mini_benchmark:constants",
        "//machina/lite/experimental/acceleration/mini_benchmark:status_codes",
        "//machina/lite/schema:schema_fbs",
        "@flatbuffers",
    ],
)

cc_test(
    name = "custom_validation_embedder_test",
    srcs = ["custom_validation_embedder_test.cc"],
    deps = [
        ":custom_validation_embedder",
        "//machina/lite:framework",
        "//machina/lite/core:framework",
        "//machina/lite/core/api:error_reporter",
        "//machina/lite/core/c:c_api_types",
        "//machina/lite/core/kernels:builtin_ops",
        "//machina/lite/experimental/acceleration/mini_benchmark:call",
        "//machina/lite/experimental/acceleration/mini_benchmark:embedded_mobilenet_model",
        "//machina/lite/experimental/acceleration/mini_benchmark:mini_benchmark_test_helper",
        "//machina/lite/experimental/acceleration/mini_benchmark:status_codes",
        "//machina/lite/kernels/internal:tensor",
        "//machina/lite/schema:schema_fbs",
        "//machina/lite/tools:model_loader",
        "@com_google_googletest//:gtest_main",
        "@flatbuffers//:runtime_cc",
    ] + libjpeg_handle_deps() + register_selected_ops_deps(),
)

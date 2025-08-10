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
load("//machina:machina.default.bzl", "get_compatible_with_portable")
load("//machina/lite:build_def.bzl", "tflite_copts", "tflite_linkopts")
load("//machina/lite:special_rules.bzl", "tflite_portable_test_suite")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

config_setting(
    name = "tflite_debug_delegate",
    define_values = {"tflite_debug_delegate": "true"},
)

cc_library(
    name = "telemetry",
    srcs = ["telemetry.cc"],
    hdrs = ["telemetry.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts(),
    deps = [
        "//machina/lite/acceleration/configuration:configuration_fbs",
        "//machina/lite/core/api",
        "//machina/lite/core/c:common",
    ],
)

cc_test(
    name = "telemetry_test",
    srcs = ["telemetry_test.cc"],
    linkopts = tflite_linkopts(),
    linkstatic = 1,
    deps = [
        ":telemetry",
        "//machina/lite/acceleration/configuration:configuration_fbs",
        "//machina/lite/core/api",
        "//machina/lite/core/c:common",
        "//machina/lite/profiling:profile_buffer",
        "@com_google_googletest//:gtest_main",
        "@flatbuffers",
    ],
)

cc_library(
    name = "utils",
    srcs = ["utils.cc"],
    hdrs = ["utils.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts(),
    deps = [
        "//machina/lite:kernel_api",
        "//machina/lite:util",
        "//machina/lite/c:c_api_types",
        "//machina/lite/c:common",
        "//machina/lite/core:subgraph",
        "//machina/lite/core/c:common",
        "//machina/lite/kernels:kernel_util",
    ],
)

cc_library(
    name = "interpreter_utils",
    srcs = ["interpreter_utils.cc"],
    hdrs = ["interpreter_utils.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts(),
    deps = [
        "//machina/lite:framework",
        "//machina/lite/c:c_api_types",
        "//machina/lite/c:common",
        "//machina/lite/core/api:error_reporter",
    ],
)

cc_test(
    name = "utils_test",
    srcs = ["utils_test.cc"],
    linkopts = tflite_linkopts(),
    linkstatic = 1,
    deps = [
        ":utils",
        "//machina/lite/core:subgraph",
        "//machina/lite/core/c:common",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "interpreter_utils_test",
    size = "small",
    srcs = ["interpreter_utils_test.cc"],
    features = ["-dynamic_link_test_srcs"],  # see go/dynamic_link_test_srcs
    deps = [
        ":delegate_test_util",
        ":interpreter_utils",
        "//machina/lite:framework",
        "//machina/lite/core/c:common",
        "//machina/lite/kernels:builtin_ops",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "delegate_test",
    size = "small",
    srcs = ["delegate_test.cc"],
    data = ["//machina/lite:testdata/add.bin"],
    features = ["-dynamic_link_test_srcs"],  # see go/dynamic_link_test_srcs.
    deps = [
        ":delegate_test_util",
        "//machina/compiler/mlir/lite/experimental/remat:metadata_util",
        "//machina/compiler/mlir/lite/schema:schema_conversion_utils",
        "//machina/lite:builtin_ops",
        "//machina/lite:framework",
        "//machina/lite:version",
        "//machina/lite/core:framework",
        "//machina/lite/core/c:c_api_experimental",
        "//machina/lite/core/c:c_api_types",
        "//machina/lite/core/c:common",
        "//machina/lite/core/kernels:builtin_ops",
        "//machina/lite/kernels:builtin_ops",
        "//machina/lite/kernels:kernel_util",
        "//machina/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest_main",
        "@flatbuffers",
    ],
)

cc_test(
    name = "opaque_delegate_test",
    size = "small",
    srcs = ["opaque_delegate_test.cc"],
    data = [
        "//machina/lite:testdata/add.bin",
        "//machina/lite:testdata/test_custom_node_with_init_data.bin",
    ],
    defines = [
        "TFLITE_USE_OPAQUE_DELEGATE",
    ],
    deps = [
        "//machina/lite:builtin_ops",
        "//machina/lite:framework",
        "//machina/lite/c:c_api_experimental",
        "//machina/lite/c:common",
        "//machina/lite/core/c:c_api_types",
        "//machina/lite/core/c:common",
        "//machina/lite/kernels:builtin_ops",
        "//machina/lite/schema:schema_fbs",
        "//machina/lite/testing:util",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "opaque_delegate_strip_error_strings_test",
    size = "small",
    srcs = ["opaque_delegate_test.cc"],
    data = [
        "//machina/lite:testdata/add.bin",
        "//machina/lite:testdata/test_custom_node_with_init_data.bin",
    ],
    defines = [
        "TFLITE_USE_OPAQUE_DELEGATE",
        "TF_LITE_STRIP_ERROR_STRINGS",
    ],
    deps = [
        "//machina/lite:builtin_ops",
        "//machina/lite:framework",
        "//machina/lite/c:c_api_experimental",
        "//machina/lite/c:common",
        "//machina/lite/core/c:c_api_types",
        "//machina/lite/core/c:common",
        "//machina/lite/kernels:builtin_ops",
        "//machina/lite/schema:schema_fbs",
        "//machina/lite/testing:util",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "delegate_test_util",
    testonly = True,
    srcs = ["delegate_test_util.cc"],
    hdrs = ["delegate_test_util.h"],
    deps = [
        ":utils",
        "//machina/lite:builtin_ops",
        "//machina/lite:util",
        "//machina/lite/core:framework",
        "//machina/lite/core/c:common",
        "//machina/lite/core/kernels:builtin_ops",
        "//machina/lite/kernels:kernel_util",
        "//machina/lite/kernels/internal:compatibility",
        "//machina/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
        "@eigen_archive//:eigen3",
    ],
)

cc_library(
    name = "serialization",
    srcs = ["serialization.cc"],
    hdrs = ["serialization.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts(),
    deps = [
        "//machina/lite:minimal_logging",
        "//machina/lite/core/c:common",
        "@farmhash_archive//:farmhash",
    ],
)

cc_test(
    name = "serialization_test",
    srcs = ["serialization_test.cc"],
    linkopts = tflite_linkopts(),
    linkstatic = 1,
    deps = [
        ":serialization",
        "//machina/lite:util",
        "//machina/lite/core/c:common",
        "@com_google_googletest//:gtest_main",
    ],
)

tflite_portable_test_suite()

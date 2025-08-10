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

# Utils package for the stable delegate APIs.
load("@bazel_skylib//rules:build_test.bzl", "build_test")
load("//machina/lite:build_def.bzl", "tflite_cc_shared_object", "tflite_linkopts_no_undefined")
load("//machina/lite/core/shims:cc_library_with_tflite.bzl", "cc_library_with_tflite")

# copybara:uncomment package(default_applicable_licenses = ["//machina:license"])

cc_library_with_tflite(
    name = "delegate_loader",
    srcs = ["delegate_loader.cc"],
    hdrs = ["delegate_loader.h"],
    generate_opaque_delegate_target = True,
    linkopts = select({
        "//machina:windows": [],
        "//conditions:default": ["-ldl"],
    }),
    tflite_deps = ["//machina/lite/acceleration/configuration/c:stable_delegate"],
    visibility = ["//visibility:public"],
    deps = [
        "//machina/lite/experimental/acceleration/compatibility:android_info",
        "//machina/lite/tools:logging",
        "@com_google_absl//absl/strings",
    ],
)

cc_library_with_tflite(
    name = "stable_delegate_interface",
    hdrs = ["stable_delegate_interface.h"],
    generate_opaque_delegate_target = True,
    tflite_deps = ["//machina/lite/acceleration/configuration/c:stable_delegate"],
    visibility = ["//visibility:public"],
)

tflite_cc_shared_object(
    name = "machinalite_stable_xnnpack_delegate",
    testonly = True,
    srcs = ["stable_xnnpack_delegate.cc"],
    defines = ["TFL_STABLE_DELEGATE_COMPILE_LIBRARY"],
    linkopts = tflite_linkopts_no_undefined() + select({
        "//machina:windows": [],
        # iOS linker doesn't support -version-script.
        "//machina:ios": [
            # TODO(b/260213307): support hiding unnecessary symbols on iOS.
        ],
        "//conditions:default": [
            # Expose necessary symbols only.
            "-Wl,--version-script,$(location //machina/lite/delegates/utils/experimental/stable_delegate:version_script.lds)",
        ],
    }),
    per_os_targets = True,
    visibility = ["//visibility:public"],
    deps = [
        ":stable_delegate_interface",
        ":version_script.lds",
        "//machina/lite/acceleration/configuration/c:stable_delegate",
        "//machina/lite/acceleration/configuration/c:xnnpack_plugin",
        "//machina/lite/c:c_api",
    ],
)

cc_test(
    name = "delegate_loader_test",
    srcs = ["delegate_loader_test.cc"],
    data = ["//machina/lite/delegates/utils/experimental/sample_stable_delegate:machinalite_sample_stable_delegate"],
    deps = [
        ":delegate_loader",
        "//machina/lite/acceleration/configuration:configuration_fbs",
        "//machina/lite/acceleration/configuration/c:stable_delegate",
        "//machina/lite/delegates/utils/experimental/sample_stable_delegate",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "tflite_settings_json_parser",
    srcs = [
        "tflite_settings_json_parser.cc",
        "//machina/lite/acceleration/configuration:configuration_fbs_contents-inl.h",
    ],
    hdrs = ["tflite_settings_json_parser.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//machina/lite/acceleration/configuration:configuration_fbs",
        "//machina/lite/kernels/internal:compatibility",
        "//machina/lite/tools:logging",
        "@flatbuffers",
    ],
)

build_test(
    name = "stable_delegate_build_test",
    targets = [
        ":delegate_loader",
        ":test_xnnpack_settings.json",
        ":machinalite_stable_xnnpack_delegate",
        ":tflite_settings_json_parser",
    ],
)

cc_test(
    name = "tflite_settings_json_parser_test",
    srcs = ["tflite_settings_json_parser_test.cc"],
    data = [
        ":test_xnnpack_settings.json",
        "//machina/lite/tools/delegates/experimental/stable_delegate:test_invalid_settings.json",
    ],
    deps = [
        ":tflite_settings_json_parser",
        "//machina/lite/acceleration/configuration:configuration_fbs",
        "@com_google_googletest//:gtest_main",
        "@flatbuffers",
    ],
)

cc_library(
    name = "kernel_test_main",
    testonly = 1,
    srcs = ["kernel_test_main.cc"],
    visibility = ["//visibility:public"],
    deps = [
        "//machina/lite/c:c_api_types",
        "//machina/lite/kernels:acceleration_test_util",
        "//machina/lite/kernels:acceleration_test_util_internal",
        "//machina/lite/kernels:test_delegate_providers_lib",
        "//machina/lite/kernels:test_util",
        "//machina/lite/kernels:test_util_delegate_providers",
        "//machina/lite/testing:util",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "stable_delegate_test_suite",
    size = "medium",
    tags = [
        "manual",
        "notap",
    ],
    deps = [
        ":kernel_test_main",
        "//machina/lite/kernels:combined_all_kernel_tests_lib",
    ],
)

exports_files(
    srcs = [
        "test_xnnpack_settings.json",
        "version_script.lds",
    ],
    visibility = ["//visibility:public"],
)

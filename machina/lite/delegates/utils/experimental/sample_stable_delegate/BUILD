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

# A sample delegate that supports addition and subtraction only.
load("//machina/lite:build_def.bzl", "tflite_cc_shared_object", "tflite_copts", "tflite_linkopts_no_undefined")
load("//machina/lite/core/shims:cc_library_with_tflite.bzl", "cc_library_with_tflite")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

cc_library_with_tflite(
    name = "sample_stable_delegate",
    testonly = True,
    srcs = [
        "sample_stable_delegate.cc",
    ],
    hdrs = [
        "sample_stable_delegate.h",
    ],
    copts = tflite_copts(),
    generate_opaque_delegate_target = True,
    tflite_deps = [
        "//machina/lite/c:c_api",
        "//machina/lite/c:c_api_experimental",
        "//machina/lite/c:c_api_types",
        "//machina/lite/c:common",
        "//machina/lite/delegates/utils:simple_opaque_delegate",
        "//machina/lite/kernels:builtin_ops",
    ],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
    ],
)

cc_test(
    name = "sample_stable_delegate_test",
    srcs = ["sample_stable_delegate_test.cc"],
    copts = tflite_copts(),
    data = [
        "//machina/lite:testdata/add.bin",
        "//machina/lite:testdata/sub.bin",
    ],
    deps = [
        ":sample_stable_delegate_opaque_delegate",
        "//machina/lite/c:c_api",
        "//machina/lite/c:c_api_experimental",
        "//machina/lite/c:c_api_types",
        "//machina/lite/c:common",
        "//machina/lite/kernels:kernel_util",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library_with_tflite(
    name = "sample_stable_delegate_external",
    testonly = True,
    srcs = [
        "sample_stable_delegate_external.cc",
    ],
    copts = tflite_copts(),
    generate_opaque_delegate_target = True,
    tflite_deps = [
        ":sample_stable_delegate",
        "//machina/lite/c:c_api",
        "//machina/lite/c:c_api_experimental",
        "//machina/lite/c:c_api_types",
        "//machina/lite/c:common",
        "//machina/lite/delegates/utils:simple_opaque_delegate",
        "//machina/lite/delegates/utils/experimental/stable_delegate:stable_delegate_interface",
        "//machina/lite/acceleration/configuration/c:delegate_plugin",
        "//machina/lite/acceleration/configuration/c:stable_delegate",
    ],
)

tflite_cc_shared_object(
    name = "machinalite_sample_stable_delegate",
    testonly = True,
    linkopts = tflite_linkopts_no_undefined() + select({
        "//machina:windows": [],
        "//conditions:default": [
            # Expose necessary symbols only.
            "-Wl,--version-script,$(location //machina/lite/delegates/utils/experimental/stable_delegate:version_script.lds)",
        ],
    }),
    per_os_targets = True,
    deps = [
        ":sample_stable_delegate_external_opaque_delegate",
        "//machina/lite/delegates/utils/experimental/stable_delegate:version_script.lds",
    ],
)

cc_test(
    name = "sample_stable_delegate_external_test",
    srcs = ["sample_stable_delegate_external_test.cc"],
    data = [
        ":machinalite_sample_stable_delegate",
        "//machina/lite:testdata/add.bin",
    ],
    deps = [
        ":sample_stable_delegate_opaque_delegate",
        "//machina/lite/acceleration/configuration:configuration_fbs",
        "//machina/lite/c:c_api",
        "//machina/lite/c:c_api_experimental",
        "//machina/lite/c:c_api_types",
        "//machina/lite/c:common",
        "//machina/lite/delegates/utils/experimental/stable_delegate:delegate_loader",
        "//machina/lite/kernels:kernel_util",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "sample_stable_delegate_kernel_tests",
    timeout = "moderate",
    args = ["--stable_delegate_settings_file=$(location stable_delegate_settings.json)"],
    data = [
        "stable_delegate_settings.json",
        ":machinalite_sample_stable_delegate",
    ],
    # This shard_count value of 30 was chosen because empirically this seems to be near the sweet
    # spot that minimizes test time while not wasting resources with unnecessary shards.
    # This runs roughly 30% faster than with shard_count = 20, but values beyond this show little
    # improvement, e.g. shard_count = 40 gave only a few per cent improvement.
    shard_count = 30,
    deps = [
        "//machina/lite/kernels:combined_all_kernel_tests_lib",
        "//machina/lite/kernels:test_main",
    ],
)

genrule(
    name = "gen_stable_delegate_settings",
    outs = ["stable_delegate_settings.json"],
    cmd = """
        echo '
            {
              "stable_delegate_loader_settings": {
                "delegate_path": "machina/lite/delegates/utils/experimental/sample_stable_delegate/libmachinalite_sample_stable_delegate.so"
              }
            }
        ' > $@
    """,
)

cc_binary(
    name = "sample_app_using_stable_delegate",
    testonly = True,
    srcs = ["sample_app_using_stable_delegate.cc"],
    data = [
        ":machinalite_sample_stable_delegate",
        "//machina/lite:testdata/add.bin",
    ],
    deps = [
        "//machina/lite:framework_stable",
        "//machina/lite/acceleration/configuration:configuration_fbs",
        "//machina/lite/c:c_api",
        "//machina/lite/delegates/utils/experimental/stable_delegate:delegate_loader",
        "//machina/lite/kernels:builtin_ops",
    ],
)

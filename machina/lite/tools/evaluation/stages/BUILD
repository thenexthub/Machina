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

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

common_linkopts = tflite_linkopts() + select({
    "//conditions:default": [],
    "//machina:android": [
        "-pie",
        "-llog",
    ],
})

exports_files(glob([
    "testdata/*.jpg",
]))

cc_library(
    name = "image_preprocessing_stage",
    srcs = ["image_preprocessing_stage.cc"],
    hdrs = ["image_preprocessing_stage.h"],
    copts = tflite_copts(),
    deps = [
        "//machina/core:tflite_portable_logging",
        "//machina/lite:string",
        "//machina/lite/c:c_api_types",
        "//machina/lite/kernels/internal:reference_base",
        "//machina/lite/kernels/internal:types",
        "//machina/lite/profiling:time",
        "//machina/lite/tools/evaluation:evaluation_stage",
        "//machina/lite/tools/evaluation/proto:evaluation_config_cc_proto",
        "//machina/lite/tools/evaluation/proto:evaluation_stages_cc_proto",
        "//machina/lite/tools/evaluation/proto:preprocessing_steps_cc_proto",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
        "@libjpeg_turbo//:jpeg",
        "@local_xla//xla/tsl/util:stats_calculator_portable",
    ] + select({
        "//machina:android": [
            "//machina/core:portable_jpeg_internal",
        ],
        "//conditions:default": [
            "//machina/core:jpeg_internal",
        ],
    }),
)

cc_test(
    name = "image_preprocessing_stage_test",
    srcs = ["image_preprocessing_stage_test.cc"],
    data = ["testdata/grace_hopper.jpg"],
    linkopts = common_linkopts,
    linkstatic = 1,
    deps = [
        ":image_preprocessing_stage",
        "//machina/lite/c:c_api_types",
        "//machina/lite/tools/evaluation/proto:evaluation_config_cc_proto",
        "//machina/lite/tools/evaluation/proto:evaluation_stages_cc_proto",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "topk_accuracy_eval_stage",
    srcs = ["topk_accuracy_eval_stage.cc"],
    hdrs = ["topk_accuracy_eval_stage.h"],
    copts = tflite_copts(),
    deps = [
        "//machina/core:tflite_portable_logging",
        "//machina/lite/c:c_api_types",
        "//machina/lite/c:common",
        "//machina/lite/tools/evaluation:evaluation_stage",
        "//machina/lite/tools/evaluation/proto:evaluation_config_cc_proto",
        "//machina/lite/tools/evaluation/proto:evaluation_stages_cc_proto",
        "@com_google_absl//absl/log",
    ],
)

cc_test(
    name = "topk_accuracy_eval_stage_test",
    srcs = ["topk_accuracy_eval_stage_test.cc"],
    linkopts = common_linkopts,
    linkstatic = 1,
    deps = [
        ":topk_accuracy_eval_stage",
        "//machina/lite/c:c_api_types",
        "//machina/lite/c:common",
        "//machina/lite/tools/evaluation/proto:evaluation_config_cc_proto",
        "//machina/lite/tools/evaluation/proto:evaluation_stages_cc_proto",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "tflite_inference_stage",
    srcs = ["tflite_inference_stage.cc"],
    hdrs = ["tflite_inference_stage.h"],
    copts = tflite_copts(),
    deps = [
        "//machina/core:tflite_portable_logging",
        "//machina/lite:framework",
        "//machina/lite/core:framework",
        "//machina/lite/core/c:common",
        "//machina/lite/core/kernels:builtin_ops",
        "//machina/lite/profiling:time",
        "//machina/lite/tools/evaluation:evaluation_delegate_provider",
        "//machina/lite/tools/evaluation:evaluation_stage",
        "//machina/lite/tools/evaluation:utils",
        "//machina/lite/tools/evaluation/proto:evaluation_config_cc_proto",
        "//machina/lite/tools/evaluation/proto:evaluation_stages_cc_proto",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log",
        "@local_xla//xla/tsl/util:stats_calculator_portable",
    ],
)

cc_test(
    name = "tflite_inference_stage_test",
    srcs = ["tflite_inference_stage_test.cc"],
    data = ["//machina/lite:testdata/add_quantized.bin"],
    linkopts = common_linkopts,
    linkstatic = 1,
    deps = [
        ":tflite_inference_stage",
        "//machina/lite:framework",
        "//machina/lite/core:framework",
        "//machina/lite/core/c:common",
        "//machina/lite/tools/evaluation:utils",
        "//machina/lite/tools/evaluation/proto:evaluation_config_cc_proto",
        "//machina/lite/tools/evaluation/proto:evaluation_stages_cc_proto",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "image_classification_stage",
    srcs = ["image_classification_stage.cc"],
    hdrs = ["image_classification_stage.h"],
    copts = tflite_copts(),
    deps = [
        ":image_preprocessing_stage",
        ":tflite_inference_stage",
        ":topk_accuracy_eval_stage",
        "//machina/core:tflite_portable_logging",
        "//machina/lite/c:c_api_types",
        "//machina/lite/tools/evaluation:evaluation_delegate_provider",
        "//machina/lite/tools/evaluation:evaluation_stage",
        "//machina/lite/tools/evaluation:utils",
        "//machina/lite/tools/evaluation/proto:evaluation_config_cc_proto",
        "//machina/lite/tools/evaluation/proto:evaluation_stages_cc_proto",
        "@com_google_absl//absl/log",
    ],
)

cc_library(
    name = "inference_profiler_stage",
    srcs = ["inference_profiler_stage.cc"],
    hdrs = ["inference_profiler_stage.h"],
    copts = tflite_copts(),
    deps = [
        ":tflite_inference_stage",
        "//machina/core:tflite_portable_logging",
        "//machina/lite/c:c_api_types",
        "//machina/lite/tools/evaluation:evaluation_delegate_provider",
        "//machina/lite/tools/evaluation:evaluation_stage",
        "//machina/lite/tools/evaluation/proto:evaluation_config_cc_proto",
        "//machina/lite/tools/evaluation/proto:evaluation_stages_cc_proto",
        "@FP16",
        "@com_google_absl//absl/log",
        "@local_xla//xla/tsl/util:stats_calculator_portable",
    ],
)

cc_test(
    name = "inference_profiler_stage_test",
    srcs = ["inference_profiler_stage_test.cc"],
    data = ["//machina/lite:testdata/add_quantized.bin"],
    linkopts = common_linkopts,
    linkstatic = 1,
    deps = [
        ":inference_profiler_stage",
        "//machina/lite/core/c:common",
        "//machina/lite/delegates/nnapi:nnapi_delegate",
        "//machina/lite/tools/evaluation/proto:evaluation_config_cc_proto",
        "//machina/lite/tools/evaluation/proto:evaluation_stages_cc_proto",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "object_detection_average_precision_stage",
    srcs = ["object_detection_average_precision_stage.cc"],
    hdrs = ["object_detection_average_precision_stage.h"],
    copts = tflite_copts(),
    deps = [
        "//machina/core:tflite_portable_logging",
        "//machina/lite/c:c_api_types",
        "//machina/lite/core/c:common",
        "//machina/lite/tools/evaluation:evaluation_stage",
        "//machina/lite/tools/evaluation/proto:evaluation_config_cc_proto",
        "//machina/lite/tools/evaluation/proto:evaluation_stages_cc_proto",
        "//machina/lite/tools/evaluation/stages/utils:image_metrics",
        "@com_google_absl//absl/log",
    ],
)

cc_test(
    name = "object_detection_average_precision_stage_test",
    srcs = ["object_detection_average_precision_stage_test.cc"],
    linkopts = common_linkopts,
    linkstatic = 1,
    deps = [
        ":object_detection_average_precision_stage",
        "//machina/lite/c:c_api_types",
        "//machina/lite/tools/evaluation/proto:evaluation_config_cc_proto",
        "//machina/lite/tools/evaluation/proto:evaluation_stages_cc_proto",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "object_detection_stage",
    srcs = ["object_detection_stage.cc"],
    hdrs = ["object_detection_stage.h"],
    copts = tflite_copts(),
    deps = [
        ":image_preprocessing_stage",
        ":object_detection_average_precision_stage",
        ":tflite_inference_stage",
        "//machina/core:tflite_portable_logging",
        "//machina/lite/c:c_api_types",
        "//machina/lite/core/c:common",
        "//machina/lite/tools/evaluation:evaluation_delegate_provider",
        "//machina/lite/tools/evaluation:evaluation_stage",
        "//machina/lite/tools/evaluation:utils",
        "//machina/lite/tools/evaluation/proto:evaluation_config_cc_proto",
        "//machina/lite/tools/evaluation/proto:evaluation_stages_cc_proto",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
    ],
)

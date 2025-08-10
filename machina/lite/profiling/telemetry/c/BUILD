load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//machina:machina.default.bzl", "get_compatible_with_portable")
load(
    "//machina/lite:build_def.bzl",
    "tflite_cc_library_with_c_headers_test",
    "tflite_copts",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

exports_files(
    srcs = [
        "profiler.h",
        "telemetry_setting.h",
    ],
    visibility = [
        "//machina/lite:__subpackages__",
    ],
)

tflite_cc_library_with_c_headers_test(
    name = "profiler",
    hdrs = ["profiler.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts(),
    deps = [":telemetry_setting"],
)

tflite_cc_library_with_c_headers_test(
    name = "telemetry_setting",
    hdrs = ["telemetry_setting.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts(),
    deps = [
        ":telemetry_setting_internal",
        "//machina/lite/core/c:common",
    ],
)

cc_library(
    name = "telemetry_setting_internal",
    srcs = ["telemetry_setting_internal.cc"],
    hdrs = ["telemetry_setting_internal.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts(),
    deps = ["//machina/lite/core/c:common"],
)

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//machina/lite:build_def.bzl", "tflite_copts")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["benchmark"],
    licenses = ["notice"],
)

package_group(
    name = "benchmark",
    packages = [
        "//machina/lite/tools/benchmark/...",
    ],
)

exports_files(
    ["benchmark_c_api.h"],
    visibility = ["//machina/lite/tools/benchmark/experimental/c:benchmark"],
)

cc_library(
    name = "benchmark_c_api",
    srcs = ["benchmark_c_api.cc"],
    hdrs = [
        "benchmark_c_api.h",
    ],
    copts = tflite_copts(),
    visibility = [
        "benchmark",
    ],
    deps = [
        "//machina/lite/c:c_api_types",
        "//machina/lite/core/c:c_api_types",
        "//machina/lite/tools/benchmark:benchmark_model_lib",
        "//machina/lite/tools/benchmark:benchmark_params",
        "//machina/lite/tools/benchmark:benchmark_tflite_model_lib",
        "@local_xla//xla/tsl/util:stats_calculator_portable",
    ],
)

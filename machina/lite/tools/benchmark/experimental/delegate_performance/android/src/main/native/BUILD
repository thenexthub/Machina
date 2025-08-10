# Description:
#  Holds the native layer of the app.

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//machina/lite/core/shims:cc_library_with_tflite.bzl", "cc_library_with_tflite", "jni_binary_with_tflite")
load("//machina/lite/tools/benchmark/experimental/delegate_performance/android:build_defs.bzl", "accuracy_benchmark_extra_deps", "latency_benchmark_extra_deps")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//machina/lite/tools/benchmark/experimental/delegate_performance/android:__subpackages__"],
    licenses = ["notice"],
)

jni_binary_with_tflite(
    name = "libdelegate_performance_benchmark.so",
    srcs = ["delegate_performance_benchmark_jni.cc"],
    tflite_deps = [":accuracy_benchmark"],
    deps = [
        ":latency_benchmark",
        "//machina/lite/acceleration/configuration:configuration_fbs",
        "//machina/lite/delegates/utils/experimental/stable_delegate:tflite_settings_json_parser",
        "//machina/lite/java/jni",
        "//machina/lite/tools/benchmark/experimental/delegate_performance/android/proto:delegate_performance_cc_proto",
        "@flatbuffers",
    ],
)

cc_library(
    name = "status_codes",
    hdrs = ["status_codes.h"],
)

cc_library(
    name = "latency_benchmark",
    srcs = ["latency_benchmark.cc"],
    hdrs = ["latency_benchmark.h"],
    deps = [
        "//machina/core/util:stats_calculator_portable",
        "//machina/lite:minimal_logging",
        "//machina/lite/acceleration/configuration:configuration_fbs",
        "//machina/lite/c:c_api_types",
        "//machina/lite/profiling:memory_info",
        "//machina/lite/tools/benchmark:benchmark_model_lib",
        "//machina/lite/tools/benchmark:benchmark_params",
        "//machina/lite/tools/benchmark:benchmark_tflite_model_lib",
        "//machina/lite/tools/benchmark/experimental/delegate_performance/android/proto:delegate_performance_cc_proto",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ] + latency_benchmark_extra_deps(),
)

cc_library_with_tflite(
    name = "accuracy_benchmark",
    srcs = ["accuracy_benchmark.cc"],
    hdrs = ["accuracy_benchmark.h"],
    deps = [
        ":status_codes",
        "//machina/lite:minimal_logging",
        "//machina/lite/acceleration/configuration:configuration_fbs",
        "//machina/lite/acceleration/configuration:gpu_plugin",
        "//machina/lite/acceleration/configuration:stable_delegate_plugin",
        "//machina/lite/acceleration/configuration:xnnpack_plugin",
        "//machina/lite/core/acceleration/configuration:nnapi_plugin",
        "//machina/lite/delegates/utils/experimental/stable_delegate:tflite_settings_json_parser",
        "//machina/lite/experimental/acceleration/mini_benchmark:libjpeg_handle_static_link",
        "//machina/lite/experimental/acceleration/mini_benchmark:status_codes",
        "//machina/lite/experimental/acceleration/mini_benchmark/c:c_api_in_process",
        "//machina/lite/kernels/internal:compatibility",
        "//machina/lite/tools:command_line_flags",
        "//machina/lite/tools:tool_params",
        "@flatbuffers",
    ] + accuracy_benchmark_extra_deps(),
)

cc_library_with_tflite(
    name = "benchmark_native",
    tflite_jni_binaries = [":libdelegate_performance_benchmark.so"],
)

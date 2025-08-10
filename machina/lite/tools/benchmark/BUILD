load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("//machina:machina.bzl", "tf_cc_binary")
load("//machina/lite:build_def.bzl", "tflite_copts", "tflite_copts_warnings", "tflite_linkopts")
load("//machina/lite:special_rules.bzl", "tflite_portable_test_suite")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

common_copts = tflite_copts() + tflite_copts_warnings()

# We create a library for benchmark_main.cc to faciliate the creation of a
# customized benchmark model binary that only needs linking with extra
# dependency, e.g., enabling creating of benchmark binaries with a custom
# delegate provider.
cc_library(
    name = "benchmark_model_main",
    srcs = [
        "benchmark_main.cc",
    ],
    copts = common_copts,
    deps = [
        ":benchmark_tflite_model_lib",
        "//machina/lite/tools:logging",
    ],
)

cc_binary(
    name = "benchmark_model",
    copts = common_copts,
    linkopts = tflite_linkopts() + select({
        "//machina:android": [
            "-pie",  # Android 5.0 and later supports only PIE
            "-lm",  # some builtin ops, e.g., tanh, need -lm
            "-Wl,--rpath=/data/local/tmp/",  # Hexagon delegate libraries should be in /data/local/tmp
        ],
        "//conditions:default": [],
    }),
    tags = ["builder_default_android_arm64"],
    deps = [
        ":benchmark_model_main",
    ],
)

cc_binary(
    name = "benchmark_model_performance_options",
    srcs = [
        "benchmark_tflite_performance_options_main.cc",
    ],
    copts = common_copts,
    linkopts = tflite_linkopts() + select({
        "//machina:android": [
            "-pie",  # Android 5.0 and later supports only PIE
            "-lm",  # some builtin ops, e.g., tanh, need -lm
            "-Wl,--rpath=/data/local/tmp/",  # Hexagon delegate libraries should be in /data/local/tmp
        ],
        "//conditions:default": [],
    }),
    tags = ["builder_default_android_arm64"],
    deps = [
        ":benchmark_performance_options",
        ":benchmark_tflite_model_lib",
        "//machina/lite/tools:logging",
    ],
)

# As with most target binaries that use flex, this should be built with the
# `--config=monolithic` build flag, e.g.,
#    bazel build --config=monolithic --config=android_arm64 \
#        -c opt --cxxopt='--std=c++17' :benchmark_model_plus_flex
tf_cc_binary(
    name = "benchmark_model_plus_flex",
    srcs = [
        "benchmark_plus_flex_main.cc",
    ],
    copts = common_copts,
    linkopts = tflite_linkopts() + select({
        "//machina:android": [
            "-pie",  # Android 5.0 and later supports only PIE
            "-lm",  # some builtin ops, e.g., tanh, need -lm
        ],
        "//conditions:default": [],
    }),
    deps = [
        ":benchmark_tflite_model_lib",
        "//machina/lite/delegates/flex:delegate",
        "//machina/lite/testing:init_machina",
        "//machina/lite/tools:logging",
    ],
)

cc_test(
    name = "benchmark_test",
    srcs = ["benchmark_test.cc"],
    args = [
        "--fp32_graph=$(location //machina/lite:testdata/multi_add.bin)",
        "--int8_graph=$(location //machina/lite:testdata/add_quantized_int8.bin)",
        "--string_graph_with_signature=$(location //machina/lite:testdata/string_input_model_with_signature.bin)",
        "--string_graph_without_signature=$(location //machina/lite:testdata/string_input_model.bin)",
        "--multi_signature_graph=$(location //machina/lite:testdata/multi_signatures.bin)",
    ],
    data = [
        "//machina/lite:testdata/add_quantized_int8.bin",
        "//machina/lite:testdata/multi_add.bin",
        "//machina/lite:testdata/multi_signatures.bin",
        "//machina/lite:testdata/string_input_model.bin",
        "//machina/lite:testdata/string_input_model_with_signature.bin",
    ],
    tags = [
        "tflite_not_portable_android",
        "tflite_not_portable_ios",
    ],
    deps = [
        ":benchmark_performance_options",
        ":benchmark_tflite_model_lib",
        "//machina/lite:framework",
        "//machina/lite:string_util",
        "//machina/lite/core/c:c_api_types",
        "//machina/lite/core/c:common",
        "//machina/lite/testing:util",
        "//machina/lite/tools:command_line_flags",
        "//machina/lite/tools:logging",
        "//machina/lite/tools/delegates:delegate_provider_hdr",
        "@com_google_absl//absl/algorithm",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "profiling_listener",
    srcs = ["profiling_listener.cc"],
    hdrs = ["profiling_listener.h"],
    copts = common_copts,
    deps = [
        ":benchmark_model_lib",
        ":benchmark_params",
        "//machina/lite:framework_stable",
        "//machina/lite/profiling:profile_summarizer",
        "//machina/lite/profiling:profile_summary_formatter",
        "//machina/lite/profiling:profiler",
        "//machina/lite/tools:logging",
    ],
)

cc_library(
    name = "benchmark_tflite_model_lib",
    srcs = ["benchmark_tflite_model.cc"],
    hdrs = ["benchmark_tflite_model.h"],
    copts = common_copts + select({
        "//machina:ios": [
            "-xobjective-c++",
        ],
        "//conditions:default": [],
    }),
    deps = [
        ":benchmark_model_lib",
        ":benchmark_params",
        ":benchmark_utils",
        ":profiling_listener",
        "//machina/core/example:example_protos_cc_impl",
        "//machina/lite:framework",
        "//machina/lite:simple_memory_arena_debug_dump",
        "//machina/lite:string_util",
        "//machina/lite/core:cc_api_stable",
        "//machina/lite/core:framework",
        "//machina/lite/core/c:c_api_types",
        "//machina/lite/core/c:common",
        "//machina/lite/core/kernels:builtin_ops",
        "//machina/lite/kernels:cpu_backend_context",
        "//machina/lite/profiling:model_runtime_info",
        "//machina/lite/profiling:profile_summary_formatter",
        "//machina/lite/profiling:profiler",
        "//machina/lite/tools:logging",
        "//machina/lite/tools:model_loader",
        "//machina/lite/tools:utils",
        "//machina/lite/tools/delegates:delegate_provider_hdr",
        "//machina/lite/tools/delegates:tflite_execution_providers",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@ruy//ruy/profiler",
    ],
)

cc_test(
    name = "benchmark_tflite_model_lib_test",
    srcs = [
        "benchmark_tflite_model_test.cc",
    ],
    data = ["@tflite_mobilenet_float//:mobilenet_v1_1.0_224.tflite"],
    tags = [
        "no_oss",  # TODO: b/361565588 - Re-enable.
    ],
    deps = [
        ":benchmark_model_lib",
        ":benchmark_params",
        ":benchmark_tflite_model_lib",
        "//machina/lite/core/c:c_api_types",
        "//machina/lite/tools:tool_params",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "benchmark_multirun_stats_recorder",
    hdrs = ["benchmark_multirun_stats_recorder.h"],
    copts = common_copts,
    deps = [":benchmark_model_lib"],
)

cc_library(
    name = "benchmark_performance_options",
    srcs = [
        "benchmark_performance_options.cc",
    ],
    hdrs = ["benchmark_performance_options.h"],
    copts = common_copts,
    deps = [
        ":benchmark_model_lib",
        ":benchmark_multirun_stats_recorder",
        ":benchmark_params",
        ":benchmark_utils",
        "//machina/core/util:stats_calculator_portable",
        "//machina/lite/core/c:c_api_types",
        "//machina/lite/core/c:common",
        "//machina/lite/delegates/hexagon:hexagon_delegate",
        "//machina/lite/nnapi:nnapi_util",
        "//machina/lite/profiling:time",
        "//machina/lite/tools:command_line_flags",
        "//machina/lite/tools:logging",
        "@com_google_absl//absl/memory",
    ] + select({
        "//machina:android": [
            "//machina/lite/delegates/gpu:delegate",
        ],
        "//conditions:default": [],
    }),
)

cc_library(
    name = "benchmark_params",
    hdrs = ["benchmark_params.h"],
    copts = common_copts,
    deps = ["//machina/lite/tools:tool_params"],
)

cc_library(
    name = "benchmark_model_lib",
    srcs = [
        "benchmark_model.cc",
    ],
    hdrs = ["benchmark_model.h"],
    copts = common_copts,
    deps = [
        ":benchmark_params",
        ":benchmark_utils",
        "//machina/core/util:stats_calculator_portable",
        "//machina/lite:framework",
        "//machina/lite/core/c:common",
        "//machina/lite/profiling:memory_info",
        "//machina/lite/profiling:memory_usage_monitor",
        "//machina/lite/profiling:time",
        "//machina/lite/tools:command_line_flags",
        "//machina/lite/tools:logging",
    ],
)

cc_library(
    name = "register_custom_op",
    srcs = [
        "register_custom_op.cc",
    ],
    hdrs = [
        "register_custom_op.h",
    ],
    copts = common_copts,
    deps = [
        "//machina/lite:op_resolver",
        "@com_google_absl//absl/base:core_headers",
    ],
    alwayslink = 1,
)

exports_files(["register_custom_op.h"])

cc_library(
    name = "benchmark_utils",
    srcs = [
        "benchmark_utils.cc",
    ],
    hdrs = ["benchmark_utils.h"],
    copts = common_copts,
    deps = ["//machina/lite/profiling:time"],
)

cc_test(
    name = "benchmark_utils_test",
    srcs = [
        "benchmark_utils_test.cc",
    ],
    deps = [
        ":benchmark_utils",
        "//machina/lite/profiling:time",
        "@com_google_googletest//:gtest_main",
    ],
)

tflite_portable_test_suite()

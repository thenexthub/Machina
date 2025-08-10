load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("@rules_cc//cc:objc_library.bzl", "objc_library")
load("//machina:machina.bzl", "clean_dep")
load("//machina:machina.default.bzl", "get_compatible_with_portable")
load("//machina/lite:build_def.bzl", "tflite_copts", "tflite_copts_warnings")
load("//machina/lite:special_rules.bzl", "tflite_portable_test_suite_combined")

# copybara:uncomment_begin(google-only)
# load("//machina/lite/experimental/perfetto_profiling:build_def.bzl", "tflite_perfetto_copts")
# copybara:uncomment_end

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

common_copts = tflite_copts() + tflite_copts_warnings()

cc_library(
    name = "profiler",
    hdrs = [
        "buffered_profiler.h",
        "noop_profiler.h",
        "profiler.h",
    ],
    compatible_with = get_compatible_with_portable(),
    copts = common_copts,
    deps = [
        ":profile_buffer",
        "//machina/lite/core/api",
    ],
)

cc_test(
    name = "profiler_test",
    srcs = ["profiler_test.cc"],
    deps = [
        ":profile_buffer",
        ":profiler",
        "//machina/lite/core/api",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "atrace_profiler",
    srcs = ["atrace_profiler.cc"],
    hdrs = ["atrace_profiler.h"],
    copts = common_copts,
    visibility = ["//visibility:private"],
    deps = [
        "//machina/lite/core/api",
    ],
)

cc_test(
    name = "atrace_profiler_test",
    srcs = ["atrace_profiler_test.cc"],
    deps = [
        ":atrace_profiler",
        "@com_google_googletest//:gtest_main",
    ],
)

objc_library(
    name = "signpost_profiler",
    hdrs = ["signpost_profiler.h"],
    copts = common_copts,
    non_arc_srcs = ["signpost_profiler.mm"],
    tags = ["apple"],
    deps = [
        "//machina/lite/core/api",
    ],
)

cc_library(
    name = "platform_profiler",
    srcs = ["platform_profiler.cc"],
    hdrs = ["platform_profiler.h"],
    compatible_with = get_compatible_with_portable(),
    copts =
        # copybara:uncomment_begin(google-only)
        # tflite_perfetto_copts() +
        # copybara:uncomment_end
        common_copts,
    deps = [
        "//machina/lite/core/api",
    ] + select({
        # copybara:uncomment_begin(google-only)
        # clean_dep("//machina/lite/experimental/perfetto_profiling:enable_tflite_perfetto_profiler_explicit_true"): [
        # "//machina/lite/experimental/perfetto_profiling:perfetto_profiler",
        # ],
        # copybara:uncomment_end
        "//machina:android": [":atrace_profiler"],
        "//machina:ios": [":signpost_profiler"],
        "//conditions:default": [],
    }),
)

cc_library(
    name = "profile_buffer",
    srcs = [
        "profile_buffer.cc",
    ],
    hdrs = ["profile_buffer.h"],
    compatible_with = get_compatible_with_portable(),
    copts = common_copts,
    deps = [
        ":memory_info",
        ":time",
        "//machina/lite:minimal_logging",
        "//machina/lite/core/api",
    ],
)

cc_test(
    name = "profile_buffer_test",
    srcs = ["profile_buffer_test.cc"],
    deps = [
        ":profile_buffer",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "time",
    srcs = ["time.cc"],
    hdrs = ["time.h"],
    compatible_with = get_compatible_with_portable(),
    copts = common_copts,
)

cc_test(
    name = "time_test",
    srcs = ["time_test.cc"],
    deps = [
        ":time",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "memory_info",
    srcs = ["memory_info.cc"],
    hdrs = ["memory_info.h"],
    compatible_with = get_compatible_with_portable(),
    copts = common_copts,
)

cc_test(
    name = "memory_info_test",
    srcs = ["memory_info_test.cc"],
    tags = [
        # Some low-level checks, like heap size check, may break in asan, msan
        # and tsan. So, disable such tests.
        "noasan",
        "nomsan",
        "notsan",
        # TODO(b/166227284): Fix the test for Android.
        "tflite_not_portable_android",
    ],
    deps = [
        ":memory_info",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "memory_usage_monitor",
    srcs = ["memory_usage_monitor.cc"],
    hdrs = ["memory_usage_monitor.h"],
    copts = common_copts,
    deps = [
        ":memory_info",
        "//machina/lite:minimal_logging",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
    ],
)

cc_test(
    name = "memory_usage_monitor_test",
    srcs = ["memory_usage_monitor_test.cc"],
    deps = [
        ":memory_info",
        ":memory_usage_monitor",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "memory_latency_logger",
    srcs = ["memory_latency_logger.cc"],
    hdrs = ["memory_latency_logger.h"],
    deps = [
        ":memory_usage_monitor",
        "//machina/lite:framework_stable",
        "//machina/lite/tools:logging",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/time",
    ],
)

cc_library(
    name = "profile_summary_formatter",
    srcs = ["profile_summary_formatter.cc"],
    hdrs = ["profile_summary_formatter.h"],
    compatible_with = get_compatible_with_portable(),
    copts = common_copts,
    deps = [
        "//machina/core/util:stats_calculator_portable",
        "//machina/lite/profiling/proto:profiling_info_cc",
        "//machina/lite/tools:logging",
    ],
)

cc_test(
    name = "profile_summary_formatter_test",
    srcs = ["profile_summary_formatter_test.cc"],
    deps = [
        ":profile_summary_formatter",
        "//machina/core/util:stats_calculator_portable",
        "//machina/lite/profiling/proto:profiling_info_cc",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "profile_summarizer",
    srcs = ["profile_summarizer.cc"],
    hdrs = ["profile_summarizer.h"],
    compatible_with = get_compatible_with_portable(),
    copts = common_copts,
    deps = [
        ":memory_info",
        ":profile_buffer",
        ":profile_summary_formatter",
        "//machina/core/util:stats_calculator_portable",
        "//machina/lite:framework",
        "//machina/lite/c:common",
        "//machina/lite/core:framework_stable",
        "//machina/lite/core/api",
    ],
)

cc_library(
    name = "root_profiler",
    srcs = ["root_profiler.cc"],
    hdrs = ["root_profiler.h"],
    compatible_with = get_compatible_with_portable(),
    copts = common_copts,
    deps = ["//machina/lite/core/api"],
)

cc_test(
    name = "root_profiler_test",
    srcs = ["root_profiler_test.cc"],
    deps = [
        ":root_profiler",
        "//machina/lite/core/api",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "subgraph_tensor_profiler",
    srcs = ["subgraph_tensor_profiler.cc"],
    hdrs = ["subgraph_tensor_profiler.h"],
    copts = common_copts,
    deps = [
        "//machina/lite:framework_stable",
        "//machina/lite/core:subgraph",
        "//machina/lite/core/api",
        "//machina/lite/core/c:common",
        "@local_tsl//tsl/platform:path",
    ],
)

cc_library(
    name = "model_runtime_info",
    srcs = ["model_runtime_info.cc"],
    hdrs = ["model_runtime_info.h"],
    copts = common_copts,
    deps = [
        "//machina/lite:framework_stable",
        "//machina/lite:optional_debug_tools",
        "//machina/lite/c:c_api_types",
        "//machina/lite/core:cc_api_stable",
        "//machina/lite/core:subgraph",
        "//machina/lite/profiling/proto:model_runtime_info_cc",
        "//machina/lite/schema:schema_fbs",
        "//machina/lite/tools:logging",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_protobuf//:protobuf",
    ],
)

cc_test(
    name = "model_runtime_info_test",
    srcs = ["model_runtime_info_test.cc"],
    deps = [
        ":model_runtime_info",
        ":profiler",
        "//machina/lite:framework",
        "//machina/lite/c:c_api_types",
        "//machina/lite/c:common",
        "//machina/lite/core/api",
        "//machina/lite/delegates/xnnpack:xnnpack_delegate_hdrs_only",
        "//machina/lite/kernels:test_util",
        "//machina/lite/profiling/proto:model_runtime_info_cc",
        "//machina/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "subgraph_tensor_profiler_test",
    srcs = ["subgraph_tensor_profiler_test.cc"],
    deps = [
        ":subgraph_tensor_profiler",
        "//machina/lite/core:subgraph",
        "//machina/lite/core/api",
        "//machina/lite/core/c:common",
        "//machina/lite/kernels:subgraph_test_util",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "profile_summarizer_test",
    srcs = ["profile_summarizer_test.cc"],
    copts = common_copts,
    deps = [
        ":profile_summarizer",
        ":profiler",
        "//machina/lite:framework",
        "//machina/lite/c:c_api_types",
        "//machina/lite/c:common",
        "//machina/lite/core:framework",
        "//machina/lite/kernels:kernel_util",
        "//machina/lite/kernels:subgraph_test_util",
        "//machina/lite/kernels:test_util",
        "//machina/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest_main",
    ],
)

tflite_portable_test_suite_combined(
    combine_conditions = {"deps": ["@com_google_googletest//:gtest_main"]},
    enable_ios_test_suite = True,
)

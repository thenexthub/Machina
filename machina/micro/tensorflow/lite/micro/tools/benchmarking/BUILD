load(
    "//machina/lite/micro:build_def.bzl",
    "tflm_cc_binary",
    "tflm_cc_library",
    "tflm_defines",
)

tflm_cc_library(
    name = "op_resolver",
    hdrs = ["op_resolver.h"],
    visibility = ["//machina/lite/micro/tools:__subpackages__"],
    deps = ["//machina/lite/micro:op_resolvers"],
)

tflm_cc_library(
    name = "metrics",
    srcs = ["metrics.cc"],
    hdrs = ["metrics.h"],
    deps = [
        "//machina/lite/kernels/internal:compatibility",
        "//machina/lite/micro:micro_log",
        "//machina/lite/micro:micro_profiler",
        "//machina/lite/micro:recording_allocators",
    ],
)

tflm_cc_library(
    name = "generic_benchmark_lib",
    srcs = ["generic_model_benchmark.cc"],
    hdrs = ["show_meta_data.h"],
    defines = tflm_defines() + ["GENERIC_BENCHMARK_NO_META_DATA"],
    deps = [
        ":metrics",
        ":op_resolver",
        "//machina/lite/c:c_api_types",
        "//machina/lite/c:common",
        "//machina/lite/micro:micro_context",
        "//machina/lite/micro:micro_log",
        "//machina/lite/micro:micro_profiler",
        "//machina/lite/micro:op_resolvers",
        "//machina/lite/micro:recording_allocators",
        "//machina/lite/micro:system_setup",
        "//machina/lite/schema:schema_fbs",
    ],
)

tflm_cc_binary(
    name = "tflm_benchmark",
    deps = [":generic_benchmark_lib"],
)

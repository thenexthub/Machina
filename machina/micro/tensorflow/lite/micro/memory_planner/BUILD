load(
    "//machina/lite/micro:build_def.bzl",
    "tflm_cc_library",
)

package(
    default_visibility = ["//visibility:public"],
    # Disabling layering_check because of http://b/177257332
    features = ["-layering_check"],
    licenses = ["notice"],
)

tflm_cc_library(
    name = "micro_memory_planner",
    hdrs = [
        "micro_memory_planner.h",
    ],
    deps = [
        "//machina/lite/c:common",
        "//machina/lite/core/api",
    ],
)

tflm_cc_library(
    name = "linear_memory_planner",
    srcs = [
        "linear_memory_planner.cc",
    ],
    hdrs = [
        "linear_memory_planner.h",
    ],
    deps = [
        ":micro_memory_planner",
        "//machina/lite/c:common",
        "//machina/lite/micro:micro_compatibility",
        "//machina/lite/micro:micro_log",
    ],
)

tflm_cc_library(
    name = "greedy_memory_planner",
    srcs = [
        "greedy_memory_planner.cc",
    ],
    hdrs = [
        "greedy_memory_planner.h",
    ],
    deps = [
        ":micro_memory_planner",
        "//machina/lite/micro:micro_compatibility",
        "//machina/lite/micro:micro_log",
    ],
)

cc_test(
    name = "linear_memory_planner_test",
    srcs = [
        "linear_memory_planner_test.cc",
    ],
    deps = [
        ":linear_memory_planner",
        "//machina/lite/micro/testing:micro_test",
    ],
)

cc_test(
    name = "greedy_memory_planner_test",
    srcs = [
        "greedy_memory_planner_test.cc",
    ],
    deps = [
        ":greedy_memory_planner",
        "//machina/lite/micro/testing:micro_test",
    ],
)

tflm_cc_library(
    name = "memory_plan_struct",
    hdrs = [
        "memory_plan_struct.h",
    ],
    deps = [
        "//machina/lite/c:common",
        "//machina/lite/core/api",
        "//machina/lite/micro:micro_compatibility",
        "//machina/lite/micro:micro_utils",
    ],
)

tflm_cc_library(
    name = "non_persistent_buffer_planner_shim",
    srcs = ["non_persistent_buffer_planner_shim.cc"],
    hdrs = [
        "non_persistent_buffer_planner_shim.h",
    ],
    deps = [
        ":memory_plan_struct",
        ":micro_memory_planner",
        "//machina/lite/c:common",
        "//machina/lite/core/api",
        "//machina/lite/micro:micro_compatibility",
    ],
)

cc_test(
    name = "non_persistent_buffer_planner_shim_test",
    srcs = [
        "non_persistent_buffer_planner_shim_test.cc",
    ],
    deps = [
        ":non_persistent_buffer_planner_shim",
        "//machina/lite/micro/testing:micro_test",
    ],
)

load(
    "//machina/lite/micro:build_def.bzl",
    "tflm_cc_library",
    "tflm_cc_test",
)

package(
    default_visibility = ["//visibility:public"],
    # Disabling layering_check because of http://b/177257332
    features = ["-layering_check"],
    licenses = ["notice"],
)

tflm_cc_library(
    name = "ibuffer_allocator",
    hdrs = [
        "ibuffer_allocator.h",
    ],
    deps = [
        "//machina/lite/c:common",
    ],
)

tflm_cc_library(
    name = "non_persistent_arena_buffer_allocator",
    srcs = ["non_persistent_arena_buffer_allocator.cc"],
    hdrs = ["non_persistent_arena_buffer_allocator.h"],
    deps = [
        ":ibuffer_allocator",
        "//machina/lite/c:common",
        "//machina/lite/micro:memory_helpers",
        "//machina/lite/micro:micro_arena_constants",
        "//machina/lite/micro:micro_compatibility",
        "//machina/lite/micro:micro_log",
    ],
)

tflm_cc_test(
    name = "non_persistent_arena_buffer_allocator_test",
    srcs = ["non_persistent_arena_buffer_allocator_test.cc"],
    deps = [
        ":non_persistent_arena_buffer_allocator",
        "//machina/lite/micro:test_helpers",
        "//machina/lite/micro/testing:micro_test",
    ],
)

tflm_cc_library(
    name = "persistent_arena_buffer_allocator",
    srcs = ["persistent_arena_buffer_allocator.cc"],
    hdrs = ["persistent_arena_buffer_allocator.h"],
    deps = [
        ":ibuffer_allocator",
        "//machina/lite/c:common",
        "//machina/lite/micro:memory_helpers",
        "//machina/lite/micro:micro_arena_constants",
        "//machina/lite/micro:micro_compatibility",
        "//machina/lite/micro:micro_log",
    ],
)

tflm_cc_test(
    name = "persistent_arena_buffer_allocator_test",
    srcs = ["persistent_arena_buffer_allocator_test.cc"],
    deps = [
        ":persistent_arena_buffer_allocator",
        "//machina/lite/micro:test_helpers",
        "//machina/lite/micro/testing:micro_test",
    ],
)

tflm_cc_library(
    name = "simple_memory_allocator",
    srcs = [
        "single_arena_buffer_allocator.cc",
    ],
    hdrs = [
        "single_arena_buffer_allocator.h",
    ],
    deps = [
        ":ibuffer_allocator",
        "//machina/lite/c:common",
        "//machina/lite/micro:memory_helpers",
        "//machina/lite/micro:micro_arena_constants",
        "//machina/lite/micro:micro_compatibility",
        "//machina/lite/micro:micro_log",
    ],
)

tflm_cc_test(
    name = "simple_memory_allocator_test",
    srcs = [
        "single_arena_buffer_allocator_test.cc",
    ],
    deps = [
        "//machina/lite/micro:micro_framework",
        "//machina/lite/micro:test_helpers",
        "//machina/lite/micro/testing:micro_test",
    ],
)

tflm_cc_library(
    name = "recording_simple_memory_allocator",
    srcs = [
        "recording_single_arena_buffer_allocator.cc",
    ],
    hdrs = [
        "recording_single_arena_buffer_allocator.h",
    ],
    deps = [
        ":simple_memory_allocator",
        "//machina/lite/kernels/internal:compatibility",
        "//machina/lite/micro:micro_compatibility",
        "//machina/lite/micro:micro_framework",
    ],
)

tflm_cc_test(
    name = "recording_simple_memory_allocator_test",
    srcs = [
        "recording_single_arena_buffer_allocator_test.cc",
    ],
    deps = [
        ":recording_simple_memory_allocator",
        "//machina/lite/micro:micro_framework",
        "//machina/lite/micro:test_helpers",
        "//machina/lite/micro/testing:micro_test",
    ],
)

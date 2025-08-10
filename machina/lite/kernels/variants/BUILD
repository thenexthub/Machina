load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("//machina:machina.default.bzl", "get_compatible_with_portable")
load("//machina/lite:build_def.bzl", "tflite_copts", "tflite_copts_warnings", "tflite_linkopts")

common_copts = tflite_copts() + tflite_copts_warnings()

# Library of list ops.
cc_library(
    name = "list_ops_lib",
    srcs = glob(
        ["list_kernels/*.cc"],
        exclude = ["list_kernels/*test*"],
    ) + ["register_list_ops.cc"],
    hdrs = [
        "list_ops_lib.h",
        "register_list_ops.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":list_ops_util",
        ":tensor_array",
        "//machina/lite:array",
        "//machina/lite:mutable_op_resolver",
        "//machina/lite:util",
        "//machina/lite/c:c_api_types",
        "//machina/lite/core/c:c_api_types",
        "//machina/lite/core/c:common",
        "//machina/lite/kernels:cpu_backend_context",
        "//machina/lite/kernels:kernel_util",
        "//machina/lite/kernels/internal:optimized_base",
        "//machina/lite/kernels/internal:runtime_shape",
        "//machina/lite/kernels/internal:tensor",
        "//machina/lite/kernels/internal:tensor_ctypes",
        "//machina/lite/schema:schema_fbs",
    ],
)

# Utils for simple kernel unit tests.
cc_library(
    name = "test_util",
    testonly = True,
    srcs = ["list_kernels/test_util.cc"],
    hdrs = ["list_kernels/test_util.h"],
    deps = [
        ":tensor_array",
        "//machina/lite:array",
        "//machina/lite:type_to_tflitetype",
        "//machina/lite:util",
        "//machina/lite/c:common",
        "//machina/lite/core/c:c_api_types",
        "//machina/lite/core/c:common",
        "//machina/lite/kernels:kernel_util",
        "//machina/lite/kernels:op_macros",
        "//machina/lite/kernels:test_util",
        "//machina/lite/schema:schema_fbs",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "list_reserve_test",
    srcs = ["list_kernels/list_reserve_test.cc"],
    deps = [
        ":list_ops_lib",
        ":tensor_array",
        "//machina/lite/core/c:c_api_types",
        "//machina/lite/core/c:common",
        "//machina/lite/kernels:test_util",
        "//machina/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "list_stack_test",
    srcs = ["list_kernels/list_stack_test.cc"],
    deps = [
        ":list_ops_lib",
        ":test_util",
        "//machina/lite/core/c:c_api_types",
        "//machina/lite/core/c:common",
        "//machina/lite/kernels:test_util",
        "//machina/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "list_set_item_test",
    srcs = ["list_kernels/list_set_item_test.cc"],
    deps = [
        ":list_ops_lib",
        ":tensor_array",
        ":test_util",
        "//machina/lite:type_to_tflitetype",
        "//machina/lite/core/c:c_api_types",
        "//machina/lite/core/c:common",
        "//machina/lite/kernels:test_util",
        "//machina/lite/kernels/internal:compatibility",
        "//machina/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "list_from_tensor_test",
    srcs = ["list_kernels/list_from_tensor_test.cc"],
    deps = [
        ":list_ops_lib",
        ":tensor_array",
        "//machina/lite/c:common",
        "//machina/lite/core/c:c_api_types",
        "//machina/lite/core/c:common",
        "//machina/lite/kernels:test_util",
        "//machina/lite/kernels/internal:compatibility",
        "//machina/lite/kernels/internal:tensor_ctypes",
        "//machina/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "list_get_item_test",
    srcs = ["list_kernels/list_get_item_test.cc"],
    deps = [
        ":list_ops_lib",
        ":test_util",
        "//machina/lite/core/c:c_api_types",
        "//machina/lite/core/c:common",
        "//machina/lite/kernels:test_util",
        "//machina/lite/kernels/internal:tensor_ctypes",
        "//machina/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "list_length_test",
    srcs = ["list_kernels/list_length_test.cc"],
    deps = [
        ":list_ops_lib",
        ":tensor_array",
        ":test_util",
        "//machina/lite/core/c:c_api_types",
        "//machina/lite/core/c:common",
        "//machina/lite/kernels:test_util",
        "//machina/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "list_element_shape_test",
    srcs = ["list_kernels/list_element_shape_test.cc"],
    deps = [
        ":list_ops_lib",
        ":tensor_array",
        ":test_util",
        "//machina/lite/core/c:c_api_types",
        "//machina/lite/core/c:common",
        "//machina/lite/kernels:test_util",
        "//machina/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "list_pop_back_test",
    srcs = ["list_kernels/list_pop_back_test.cc"],
    deps = [
        ":list_ops_lib",
        ":tensor_array",
        ":test_util",
        "//machina/lite/core/c:c_api_types",
        "//machina/lite/core/c:common",
        "//machina/lite/kernels:test_util",
        "//machina/lite/kernels/internal:tensor_ctypes",
        "//machina/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "list_push_back_test",
    srcs = ["list_kernels/list_push_back_test.cc"],
    deps = [
        ":list_ops_lib",
        ":tensor_array",
        ":test_util",
        "//machina/lite:type_to_tflitetype",
        "//machina/lite/core/c:c_api_types",
        "//machina/lite/core/c:common",
        "//machina/lite/kernels:kernel_util",
        "//machina/lite/kernels:test_util",
        "//machina/lite/kernels/internal:compatibility",
        "//machina/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "variant_add_n_test",
    srcs = ["list_kernels/variant_add_n_test.cc"],
    deps = [
        ":list_ops_lib",
        ":tensor_array",
        ":test_util",
        "//machina/lite:type_to_tflitetype",
        "//machina/lite/core/c:c_api_types",
        "//machina/lite/core/c:common",
        "//machina/lite/kernels:kernel_util",
        "//machina/lite/kernels:test_util",
        "//machina/lite/kernels/internal:compatibility",
        "//machina/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "variant_zeros_like_test",
    srcs = ["list_kernels/variant_zeros_like_test.cc"],
    deps = [
        ":list_ops_lib",
        ":tensor_array",
        ":test_util",
        "//machina/lite/core/c:c_api_types",
        "//machina/lite/core/c:common",
        "//machina/lite/kernels:kernel_util",
        "//machina/lite/kernels:test_util",
        "//machina/lite/kernels/internal:compatibility",
        "//machina/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "list_ops_util",
    srcs = ["list_ops_util.cc"],
    hdrs = ["list_ops_util.h"],
    visibility = ["//machina/lite/kernels/variants:__subpackages__"],
    deps = [
        ":tensor_array",
        "//machina/lite:array",
        "//machina/lite:util",
        "//machina/lite/c:c_api_types",
        "//machina/lite/core/c:common",
    ],
)

cc_test(
    name = "list_ops_util_test",
    srcs = ["list_ops_util_test.cc"],
    deps = [
        ":list_ops_util",
        ":tensor_array",
        "//machina/lite:array",
        "//machina/lite:util",
        "//machina/lite/core/c:c_api_types",
        "//machina/lite/core/c:common",
        "//machina/lite/kernels:test_util",
        "@com_google_googletest//:gtest_main",
    ],
)

# Utils for testing complex multi-op subgraphs.
cc_library(
    name = "list_ops_subgraph_test_util",
    testonly = True,
    srcs = ["list_ops_subgraph_test_util.cc"],
    hdrs = ["list_ops_subgraph_test_util.h"],
    deps = [
        ":list_ops_lib",
        "//machina/lite:builtin_ops",
        "//machina/lite/c:c_api_types",
        "//machina/lite/c:common",
        "//machina/lite/core:cc_api_stable",
        "//machina/lite/core:subgraph",
        "//machina/lite/core/c:common",
        "//machina/lite/kernels:builtin_ops",
        "//machina/lite/kernels:op_macros",
        "//machina/lite/kernels:subgraph_test_util",
        "//machina/lite/schema:schema_fbs",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "list_ops_subgraph_test",
    srcs = ["list_ops_subgraph_test.cc"],
    deps = [
        ":list_ops_subgraph_test_util",
        ":tensor_array",
        "//machina/lite:array",
        "//machina/lite:interpreter_test_util",
        "//machina/lite:util",
        "//machina/lite/c:c_api_types",
        "//machina/lite/c:common",
        "//machina/lite/kernels:kernel_util",
        "//machina/lite/kernels:op_macros",
        "//machina/lite/kernels:test_util",
        "//machina/lite/kernels/internal:compatibility",
        "//machina/lite/schema:schema_fbs",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

# Type erased class that sits behind the `data.data` of variant tensors.
cc_library(
    name = "tensor_array",
    srcs = ["tensor_array.cc"],
    hdrs = ["tensor_array.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = ["//visibility:private"],
    deps = [
        "//machina/lite:array",
        "//machina/lite:util",
        "//machina/lite/c:common",
        "//machina/lite/core/c:common",
    ],
)

cc_test(
    name = "tensor_array_test",
    srcs = ["tensor_array_test.cc"],
    deps = [
        ":tensor_array",
        "//machina/lite:array",
        "//machina/lite:type_to_tflitetype",
        "//machina/lite:util",
        "//machina/lite/c:c_api_types",
        "//machina/lite/c:common",
        "//machina/lite/core/c:c_api_types",
        "//machina/lite/core/c:common",
        "//machina/lite/kernels:test_util",
        "@com_google_googletest//:gtest_main",
    ],
)

# Benchmark tool override with registered custom tensorlist ops.
cc_library(
    name = "register_list_ops_for_benchmark",
    srcs = ["register_list_ops.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":list_ops_lib",
        "//machina/lite:mutable_op_resolver",
        "//machina/lite:op_resolver",
        "//machina/lite/tools/benchmark:register_custom_op",
    ],
    alwayslink = 1,
)

cc_binary(
    name = "benchmark_model_with_list_ops",
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
        ":register_list_ops_for_benchmark",
        "//machina/lite/tools/benchmark:benchmark_model_main",
    ],
)

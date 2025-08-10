load(":tests.bzl", "hexagon_op_tests")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "hexagon_delegate_op_model",
    testonly = 1,
    hdrs = ["hexagon_delegate_op_model.h"],
    deps = [
        "//machina/lite:framework",
        "//machina/lite/core:framework",
        "//machina/lite/core/c:common",
        "//machina/lite/core/kernels:builtin_ops",
        "//machina/lite/delegates/hexagon:hexagon_delegate",
        "//machina/lite/kernels:test_util",
        "//machina/lite/kernels/internal:reference_base",
        "//machina/lite/kernels/internal:tensor",
        "//machina/lite/kernels/internal:types",
        "//machina/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
    ],
)

hexagon_op_tests(
    srcs = [
        "activations_test.cc",
        "arg_min_max_test.cc",
        "arithmetic_test.cc",
        "concat_test.cc",
        "conv_test.cc",
        "l2_norm_test.cc",
        "matmul_test.cc",
        "min_max_builder_test.cc",
        "mirror_pad_test.cc",
        "mul_test.cc",
        "neg_test.cc",
        "pack_test.cc",
        "pad_test.cc",
        "pool_test.cc",
        "quantize_test.cc",
        "reduce_test.cc",
        "reshape_test.cc",
        "resize_test.cc",
        "rsqrt_test.cc",
        "slice_test.cc",
        "softmax_test.cc",
        "space_to_depth_test.cc",
        "split_test.cc",
        "squared_difference_test.cc",
        "strided_slice_test.cc",
        "transpose_conv_test.cc",
        "transpose_test.cc",
    ],
    deps = [
        ":hexagon_delegate_op_model",
        "//machina/lite/core/c:common",
        "//machina/lite/kernels:kernel_util",
        "//machina/lite/kernels:reshape_test_common",
        "//machina/lite/kernels/internal:test_util",
        "//machina/lite/kernels/internal:types",
        "//machina/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "batch_seq_config_test",
    srcs = [
        "batch_seq_config_test.cc",
    ],
    tags = [
        "no_oss",
        "nobuilder",
        "notap",
    ],
    deps = [
        "//machina/core:framework_internal",
        "//machina/lite:framework",
        "//machina/lite:string",
        "//machina/lite/c:common",
        "//machina/lite/core/api:op_resolver",
        "//machina/lite/core/c:common",
        "//machina/lite/delegates/hexagon:hexagon_delegate",
        "//machina/lite/kernels:builtin_ops",
        "//machina/lite/kernels:kernel_util",
        "//machina/lite/testing:util",
        "//machina/lite/tools:logging",
        "//machina/lite/tools/benchmark:benchmark_utils",
        "@com_google_googletest//:gtest",
    ],
)

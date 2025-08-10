load(
    "@local_xla//xla/tsl/mkl:build_defs.bzl",
    "mkl_deps",
)
load(
    "//machina:machina.bzl",
    "tf_cc_test_mkl",
    "tf_mkl_kernel_library",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = [
        "//machina:__subpackages__",
        "//machina:internal",
    ],
    licenses = ["notice"],
)

# Public support libraries ----------------------------------------------------
MKL_SHORT_DEPS = [
    "//machina/core:core_cpu",
    "//machina/core:framework",
    "//machina/core:lib",
    "//machina/core:lib_internal",
    "//machina/core/framework:bounds_check",
    "//machina/core/kernels:ops_util",
    "//machina/core/util:onednn_env_vars",
] + mkl_deps()

MKL_DEPS = MKL_SHORT_DEPS + [
    "@eigen_archive//:eigen3",
    "//machina/core:array_grad",
    "//machina/core:math_grad",
    "//machina/core:nn_grad",
    "//machina/core:protos_all_cc",
    "//machina/core/kernels:concat_lib",
    "//machina/core/kernels:conv_2d",
    "@local_xla//xla/tsl/framework/contraction:eigen_contraction_kernel",
    "//machina/core/kernels:fill_functor",
    "//machina/core/kernels:gather_functor",
    "//machina/core/kernels:transpose_functor",
]

MKL_TEST_DEPS = [
    "//machina/cc:cc_ops",
    "//machina/core:core_cpu",
    "//machina/core:framework",
    "//machina/core:framework_internal",
    "//machina/core:lib",
    "//machina/core:protos_all_cc",
    "//machina/core:test",
    "//machina/core:test_main",
    "//machina/core:testlib",
    "//machina/core/kernels:ops_testutil",
    "//machina/core/kernels:ops_util",
    "//machina/core/util:onednn_env_vars",
] + mkl_deps()

tf_mkl_kernel_library(
    name = "mkl_batch_matmul_op",
    srcs = ["mkl_batch_matmul_op.cc"],
    hdrs = [
        "mkl_batch_matmul_helper.h",
        "mkl_kernel_util.h",
        "mkl_matmul_ops_common.h",
    ],
    deps = [
        "//machina/core:graph",
        "//machina/core/kernels:batch_matmul_op",
        "//machina/core/kernels:matmul_op",
    ] + MKL_DEPS,
)

tf_mkl_kernel_library(
    name = "mkl_einsum_op",
    srcs = ["mkl_einsum_op.cc"],
    hdrs = [
        "mkl_batch_matmul_helper.h",
        "mkl_kernel_util.h",
        "mkl_matmul_ops_common.h",
    ],
    deps = [
        "//machina/core:graph",
        "//machina/core/kernels/linalg:einsum_op",
    ] + MKL_DEPS,
)

tf_mkl_kernel_library(
    name = "mkl_matmul_op",
    srcs = [
        "mkl_matmul_op.cc",
        "mkl_matmul_op_fused.cc",
    ],
    hdrs = [
        "mkl_kernel_util.h",
        "mkl_matmul_ops_common.h",
        "mkl_quantized_conv_ops.h",
    ],
    deps = [
        "//machina/core:graph",
        "@com_google_absl//absl/container:inlined_vector",
    ] + MKL_DEPS,
)

tf_mkl_kernel_library(
    name = "mkl_sparse_matrix_matmul_op",
    srcs = [
        "mkl_sparse_matrix_matmul_op.cc",
    ],
    hdrs = [
        "mkl_kernel_util.h",
        "mkl_matmul_ops_common.h",
    ],
    deps = [
        "//machina/core:graph",
        "//machina/core/kernels:cwise_op",
        "//machina/core/kernels:dense_update_functor",
        "//machina/core/kernels/sparse:kernels",
        "//machina/core/kernels/sparse:sparse_matrix",
    ] + MKL_DEPS,
)

tf_cc_test_mkl(
    name = "mkl_sparse_matrix_matmul_op_benchmark",
    size = "small",
    srcs = ["mkl_sparse_matrix_matmul_op_benchmark.cc"],
    linkstatic = 1,
    deps = [
        "//machina/core/kernels/mkl:mkl_sparse_matrix_matmul_op",
    ] + MKL_TEST_DEPS,
)

tf_cc_test_mkl(
    name = "mkl_quantized_conv_ops_perchannel_test",
    size = "small",
    srcs = ["mkl_quantized_conv_ops_perchannel_test.cc"],
    linkstatic = 1,  # Fixes dyld error on MacOS.
    deps = [
        ":mkl_conv_op",
        "//machina/core:array_ops_op_lib",
        "//machina/core:math_ops_op_lib",
        "//machina/core:nn_ops_op_lib",
        "//machina/core/kernels:quantization_utils",
        "//machina/core/kernels:quantized_ops",
    ] + MKL_TEST_DEPS,
)

tf_cc_test_mkl(
    name = "mkl_quantized_conv_ops_test",
    size = "small",
    srcs = ["mkl_quantized_conv_ops_test.cc"],
    linkstatic = 1,  # Fixes dyld error on MacOS.
    deps = [
        ":mkl_conv_op",
        ":mkl_kernel_util",
        "//machina/core:array_ops_op_lib",
        "//machina/core:direct_session",
        "//machina/core:math_ops_op_lib",
        "//machina/core:nn_ops_op_lib",
        "//machina/core/kernels:bias_op",
        "//machina/core/kernels:conv_ops",
        "//machina/core/kernels:depthwise_conv_op",
        "//machina/core/kernels:quantization_utils",
        "//machina/core/kernels:quantized_ops",
        "//machina/core/kernels:relu_op",
    ] + MKL_TEST_DEPS,
)

tf_cc_test_mkl(
    name = "mkl_qmatmul_op_test",
    size = "small",
    srcs = ["mkl_qmatmul_op_test.cc"],
    linkstatic = 1,  # Fixes dyld error on MacOS.
    deps = [
        ":mkl_matmul_op",
        ":mkl_qmatmul_op",
        "//machina/core:array_ops_op_lib",
        "//machina/core:math_ops_op_lib",
        "//machina/core:nn_ops_op_lib",
        "//machina/core/framework:fake_input",
        "//machina/core/framework:tensor_testutil",
        "//machina/core/kernels:quantization_utils",
        "//machina/core/kernels:quantized_ops",
    ] + MKL_TEST_DEPS,
)

tf_mkl_kernel_library(
    name = "mkl_quantize_op",
    srcs = ["mkl_quantize_op.cc"],
    deps = [
        "//machina/core/graph:mkl_graph_util",
        "//machina/core/kernels:quantized_ops",
        "@gemmlowp",
    ] + MKL_DEPS,
)

tf_cc_test_mkl(
    name = "mkl_quantize_op_test",
    size = "small",
    srcs = ["mkl_quantize_op_test.cc"],
    linkstatic = 1,  # Fixes dyld error on MacOS.
    deps = [
        ":mkl_quantize_op",
        "//machina/core:array_ops_op_lib",
        "//machina/core:math_ops_op_lib",
        "//machina/core:nn_ops_op_lib",
    ] + MKL_TEST_DEPS,
)

tf_cc_test_mkl(
    name = "mkl_quantized_pooling_ops_test",
    size = "small",
    srcs = ["mkl_quantized_pooling_ops_test.cc"],
    linkstatic = 1,  # Fixes dyld error on MacOS.
    deps = [
        ":mkl_kernel_util",
        ":mkl_pooling_ops",
        "//machina/core:array_ops_op_lib",
        "//machina/core:math_ops_op_lib",
        "//machina/core:nn_ops_op_lib",
        "//machina/core/kernels:quantization_utils",
        "//machina/core/kernels:quantized_ops",
    ] + MKL_TEST_DEPS,
)

tf_cc_test_mkl(
    name = "mkl_quantized_concat_op_test",
    size = "small",
    srcs = ["mkl_quantized_concat_op_test.cc"],
    linkstatic = 1,  # Fixes dyld error on MacOS.
    deps = [
        ":mkl_concat_op",
        "//machina/core:array_ops_op_lib",
        "//machina/core:math_ops_op_lib",
        "//machina/core:mkl_array_ops_op_lib",
        "//machina/core:nn_ops_op_lib",
        "//machina/core/kernels:quantization_utils",
        "//machina/core/kernels:quantized_ops",
    ] + MKL_TEST_DEPS,
)

tf_mkl_kernel_library(
    name = "mkl_qmatmul_op",
    srcs = ["mkl_qmatmul_op.cc"],
    hdrs = [
        "mkl_kernel_util.h",
        "mkl_matmul_ops_common.h",
        "mkl_quantized_conv_ops.h",
    ],
    deps = [
        "//machina/core:graph",
        "//machina/core:math_ops_op_lib",
        "//machina/core:mkl_nn_ops_op_lib",
        "//machina/core:nn_ops_op_lib",
        "//machina/core/kernels:matmul_op",
        "//machina/core/kernels:no_op",
    ] + MKL_DEPS,
)

tf_mkl_kernel_library(
    name = "mkl_conv_op",
    hdrs = [
        "mkl_kernel_util.h",
        "mkl_quantized_conv_ops.h",
    ],
    prefix = "mkl_conv",
    deps = [
        "//machina/core:graph",
        "//machina/core/kernels:conv_grad_shape_utils",
        "//machina/core/kernels:conv_ops",
        "//machina/core/kernels:no_op",
        "@com_google_absl//absl/strings",
    ] + MKL_DEPS,
)

tf_cc_test_mkl(
    name = "mkl_conv_ops_test",
    size = "small",
    srcs = ["mkl_conv_ops_test.cc"],
    linkstatic = 1,  # Fixes dyld error on MacOS.
    deps = MKL_TEST_DEPS,
)

tf_cc_test_mkl(
    name = "mkl_relu_op_test",
    size = "small",
    srcs = ["mkl_relu_op_test.cc"],
    linkstatic = 1,  # Fixes dyld error on MacOS.
    deps = ["@com_google_absl//absl/strings"] + MKL_TEST_DEPS,
)

tf_cc_test_mkl(
    name = "mkl_swish_op_test",
    size = "small",
    srcs = ["mkl_swish_op_test.cc"],
    linkstatic = 1,  # Fixes dyld error on MacOS.
    deps = [
        ":mkl_eltwise_activation_base_op",
        ":mkl_swish_op",
        "//machina/cc:math_ops",
        "//machina/core:direct_session",
        "//machina/core/kernels:cwise_op",
        "@com_google_absl//absl/strings",
    ] + MKL_TEST_DEPS,
)

tf_mkl_kernel_library(
    name = "mkl_tfconv_op",
    prefix = "mkl_tfconv",
    deps = MKL_SHORT_DEPS,
)

tf_mkl_kernel_library(
    name = "mkl_pooling_ops",
    srcs = [
        "mkl_avgpooling_op.cc",
        "mkl_maxpooling_op.cc",
        "mkl_pooling_ops_common.cc",
    ],
    hdrs = ["mkl_pooling_ops_common.h"],
    deps = MKL_SHORT_DEPS + ["//machina/core/kernels:pooling_ops_hdrs"],
)

tf_mkl_kernel_library(
    name = "mkl_dequantize_op",
    srcs = ["mkl_dequantize_op.cc"],
    deps = [
        "//machina/core:array_ops_op_lib",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:math_ops_op_lib",
        "//machina/core:nn_ops_op_lib",
        "//machina/core/graph:mkl_graph_util",
        "//machina/core/kernels:concat_lib_hdrs",
        "//machina/core/kernels:conv_ops",
        "//machina/core/kernels:cwise_op",
        "//machina/core/kernels:eigen_helpers",
        "//machina/core/kernels:meta_support",
        "//machina/core/kernels:pooling_ops",
        "//machina/core/kernels:quantization_utils",
        "//machina/core/kernels:quantized_ops",
        "//machina/core/kernels:transpose_functor",
        "//machina/core/util:image_resizer_state",
        "//machina/core/util:onednn_env_vars",
        "@gemmlowp",
    ] + mkl_deps(),
)

tf_cc_test_mkl(
    name = "mkl_dequantize_op_test",
    size = "small",
    srcs = ["mkl_dequantize_op_test.cc"],
    linkstatic = 1,  # Fixes dyld error on MacOS.
    deps = [
        ":mkl_dequantize_op",
        ":mkl_kernel_util",
        ":mkl_tfconv_op",
        "//machina/core:array_ops_op_lib",
        "//machina/core:math_ops_op_lib",
        "//machina/core:mkl_array_ops_op_lib",
        "//machina/core:nn_ops_op_lib",
    ] + MKL_TEST_DEPS,
)

tf_mkl_kernel_library(
    name = "mkl_relu_op",
    prefix = "mkl_relu",
    deps = MKL_DEPS,
)

tf_mkl_kernel_library(
    name = "mkl_eltwise_activation_base_op",
    prefix = "mkl_eltwise_activation_base",
    deps = MKL_DEPS,
)

tf_mkl_kernel_library(
    name = "mkl_swish_op",
    hdrs = ["mkl_eltwise_activation_base_op.h"],
    prefix = "mkl_swish",
    deps = MKL_DEPS,
)

tf_mkl_kernel_library(
    name = "mkl_fused_mish_op",
    hdrs = ["mkl_eltwise_activation_base_op.h"],
    prefix = "mkl_fused_mish",
    deps = MKL_DEPS,
)

tf_mkl_kernel_library(
    name = "mkl_softmax_op",
    prefix = "mkl_softmax",
    deps = [
        "//machina/core/graph:mkl_graph_util",
        "@eigen_archive//:eigen3",
    ] + MKL_SHORT_DEPS,
)

tf_mkl_kernel_library(
    name = "mkl_tmp_ops",
    prefix = "mkl_tmp_ops",
    deps = MKL_DEPS,
)

tf_mkl_kernel_library(
    name = "mkl_fused_batch_norm_op",
    srcs = ["mkl_fused_batch_norm_op.cc"],
    deps = [
        "//machina/core/kernels:fused_batch_norm_op",
        "//machina/core/kernels:no_op",
    ] + MKL_DEPS,
)

tf_mkl_kernel_library(
    name = "mkl_layer_norm_op",
    srcs = ["mkl_layer_norm_op.cc"],
    deps = [
        "//machina/core:mkl_nn_ops_op_lib",
        "//machina/core:nn_ops_op_lib",
    ] + MKL_DEPS,
)

tf_cc_test_mkl(
    name = "mkl_fused_batch_norm_op_test",
    size = "small",
    srcs = ["mkl_fused_batch_norm_op_test.cc"],
    linkstatic = 1,
    deps = [
        ":mkl_conv_op",
        ":mkl_fused_batch_norm_op",
        "//machina/core:direct_session",
        "//machina/core/graph:mkl_graph_util",
        "//machina/core/kernels:conv_ops_gpu_hdrs",
    ] + MKL_TEST_DEPS,
)

tf_mkl_kernel_library(
    name = "mkl_fused_instance_norm_op",
    srcs = ["mkl_fused_instance_norm_op.cc"],
    deps = [
        "//machina/core:mkl_nn_ops_op_lib",
        "//machina/core:nn_ops_op_lib",
    ] + MKL_DEPS,
)

tf_mkl_kernel_library(
    name = "mkl_concat_op",
    prefix = "mkl_concat_op",
    deps = [
        "//machina/core/kernels:no_op",
        "//machina/core/kernels:quantization_utils",
    ] + MKL_DEPS,
)

tf_mkl_kernel_library(
    name = "mkl_requantize_ops",
    srcs = [
        "mkl_requantization_range_per_channel_op.cc",
        "mkl_requantize_per_channel_op.cc",
    ],
    deps = [
        "//machina/core/kernels:concat_lib_hdrs",
        "//machina/core/kernels:conv_ops",
        "//machina/core/kernels:eigen_helpers",
        "//machina/core/kernels:meta_support",
        "//machina/core/kernels:no_op",
        "//machina/core/kernels:pooling_ops",
        "//machina/core/kernels:quantization_utils",
        "//machina/core/util:image_resizer_state",
        "@gemmlowp",
    ] + MKL_DEPS,
)

tf_cc_test_mkl(
    name = "mkl_requantize_ops_test",
    size = "small",
    srcs = ["mkl_requantize_ops_test.cc"],
    linkstatic = 1,  # Fixes dyld error on MacOS.
    deps = [
        ":mkl_requantize_ops",
        "//machina/core:array_ops_op_lib",
        "//machina/core:math_ops_op_lib",
        "//machina/core/kernels:quantization_utils",
        "//machina/core/kernels:quantized_ops",
    ] + MKL_TEST_DEPS,
)

tf_cc_test_mkl(
    name = "mkl_fused_ops_test",
    size = "small",
    srcs = ["mkl_fused_ops_test.cc"],
    linkstatic = 1,
    deps = [
        ":mkl_conv_op",
        ":mkl_matmul_op",
        ":mkl_tfconv_op",
        "//machina/cc:cc_ops_internal",
        "//machina/core:direct_session",
        "//machina/core:machina",
        "//machina/core/graph:mkl_graph_util",
        "//machina/core/kernels:bias_op",
        "//machina/core/kernels:conv_ops",
        "//machina/core/kernels:depthwise_conv_op",
        "//machina/core/kernels:matmul_op",
        "//machina/core/kernels:pad_op",
        "//machina/core/kernels:relu_op",
        "//machina/core/kernels/image",
    ] + MKL_TEST_DEPS,
)

tf_mkl_kernel_library(
    name = "mkl_transpose_op",
    srcs = [
        "mkl_transpose_op.cc",
    ],
    deps = MKL_DEPS + ["//machina/core/kernels:transpose_op"],
)

tf_cc_test_mkl(
    name = "mkl_matmul_op_benchmark",
    size = "small",
    srcs = ["mkl_matmul_op_benchmark.cc"],
    linkstatic = 1,
    deps = [
        "//machina/core/kernels:matmul_op",
        "//machina/core/kernels/mkl:mkl_batch_matmul_op",
        "//machina/core/kernels/mkl:mkl_matmul_op",
    ] + MKL_TEST_DEPS,
)

tf_mkl_kernel_library(
    name = "mkl_kernel_util",
    srcs = ["mkl_kernel_util.cc"],
    hdrs = ["mkl_kernel_util.h"],
    deps = MKL_DEPS + [
        ":mkl_quantize_op",
        "//machina/cc:cc_ops",
        "//machina/cc:ops",
        "//machina/cc:scope",
        "//machina/core:core_cpu_base",
        "//machina/core:direct_session",
        "//machina/core:graph",
        "//machina/core/kernels:quantization_utils",
    ],
)

tf_cc_test_mkl(
    name = "onednn_nn_ops_benchmark",
    size = "small",
    srcs = ["onednn_nn_ops_benchmark.cc"],
    linkstatic = 1,
    deps = [
        "//machina/cc:cc_ops_internal",
        "//machina/core/kernels:softmax_op",
        "//machina/core/kernels/mkl:mkl_softmax_op",
    ] + MKL_TEST_DEPS,
)

tf_cc_test_mkl(
    name = "onednn_fused_matmul_ops_test",
    size = "medium",
    srcs = ["onednn_fused_matmul_ops_test.cc"],
    linkstatic = 1,  # Fixes dyld error on MacOS.
    deps = [
        ":mkl_kernel_util",
        ":mkl_matmul_op",
        "//machina/cc:cc_ops_internal",
        "//machina/core:direct_session",
        "//machina/core/kernels:bias_op",
        "//machina/core/kernels:matmul_op",
        "//machina/core/kernels:quantization_utils",
        "//machina/core/kernels:relu_op",
        "@com_google_absl//absl/strings",
    ] + MKL_TEST_DEPS,
)

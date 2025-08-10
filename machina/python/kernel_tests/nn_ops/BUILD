# Tests of TensorFlow NN kernels written using the Python API.

load("//machina:strict.default.bzl", "py_strict_library")
load("//machina:machina.default.bzl", "cuda_py_strict_test", "tf_py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//machina:internal"],
    licenses = ["notice"],
)

CONV_TEST_BASENAMES = [
    ":atrous_conv2d_test",
    ":conv1d_test",
    ":conv1d_transpose_test",
    ":conv2d_backprop_filter_grad_test",
    ":conv3d_transpose_test",
    ":conv3d_backprop_filter_v2_grad_test",
    ":conv_ops_3d_test",
    ":conv_ops_test",
    ":depthwise_conv_op_test",
    ":conv2d_transpose_test",
]

test_suite(
    name = "conv_tests",
    tests = [basename for basename in CONV_TEST_BASENAMES] +
            [basename + "_gpu" for basename in CONV_TEST_BASENAMES],
)

cuda_py_strict_test(
    name = "atrous_conv2d_test",
    size = "medium",
    srcs = ["atrous_conv2d_test.py"],
    shard_count = 2,
    tags = [
        "no_gpu",  #  Flaky: b/80127739, b/127001953
    ],
    deps = [
        "//machina/python/framework:constant_op",
        "//machina/python/framework:errors",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:gradient_checker",
        "//machina/python/ops:nn_grad",
        "//machina/python/ops:nn_impl",
        "//machina/python/ops:nn_ops",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "atrous_convolution_test",
    size = "medium",
    srcs = ["atrous_convolution_test.py"],
    tags = [
        "manual",
        "no_cuda_asan",
    ],
    deps = [
        "//machina/python/eager:context",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:gradient_checker",
        "//machina/python/ops:nn_grad",
        "//machina/python/ops:nn_ops",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "betainc_op_test",
    size = "small",
    srcs = ["betainc_op_test.py"],
    xla_tags = [
        "no_cuda_asan",  # times out
    ],
    deps = [
        "//machina/python/framework:constant_op",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:gradient_checker",
        "//machina/python/ops:gradients_impl",
        "//machina/python/ops:math_ops",
        "//machina/python/platform:client_testlib",
        "//machina/python/platform:tf_logging",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "bias_op_base",
    srcs = ["bias_op_base.py"],
    deps = [
        "//machina/python/eager:backprop",
        "//machina/python/eager:context",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:errors",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:gradient_checker",
        "//machina/python/ops:gradient_checker_v2",
        "//machina/python/ops:gradients_impl",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:nn_grad",
        "//machina/python/ops:nn_ops",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "bias_op_d9m_test",
    size = "medium",
    srcs = ["bias_op_d9m_test.py"],
    shard_count = 2,
    deps = [
        ":bias_op_base",
        "//machina/python/eager:backprop",
        "//machina/python/eager:context",
        "//machina/python/framework:config",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:gradients_impl",
        "//machina/python/ops:nn_ops",
        "//machina/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_strict_test(
    name = "bias_op_test",
    size = "medium",
    srcs = ["bias_op_test.py"],
    deps = [
        ":bias_op_base",
        "//machina/python/platform:client_testlib",
    ],
)

cuda_py_strict_test(
    name = "conv1d_test",
    size = "small",
    srcs = ["conv1d_test.py"],
    deps = [
        "//machina/python/framework:constant_op",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:array_ops_stack",
        "//machina/python/ops:nn_ops",
        "//machina/python/platform:client_testlib",
    ],
)

cuda_py_strict_test(
    name = "conv1d_transpose_test",
    size = "small",
    srcs = ["conv1d_transpose_test.py"],
    deps = [
        "//machina/python/framework:constant_op",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:gradient_checker",
        "//machina/python/ops:nn_grad",
        "//machina/python/ops:nn_ops",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "conv2d_backprop_filter_grad_test",
    size = "medium",
    srcs = ["conv2d_backprop_filter_grad_test.py"],
    shard_count = 2,
    tags = [
        "optonly",  # flaky timeouts unless optimized
    ],
    deps = [
        "//machina/python/framework:constant_op",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:gradient_checker",
        "//machina/python/ops:nn_grad",
        "//machina/python/ops:nn_ops",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "conv2d_transpose_test",
    size = "small",
    srcs = ["conv2d_transpose_test.py"],

    # TODO(b/144432983): S32 convolutions should not be auto-clustered, only
    # crashes tests.
    xla_enable_strict_auto_jit = False,
    deps = [
        "//machina/python/framework:constant_op",
        "//machina/python/framework:errors",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:array_ops_stack",
        "//machina/python/ops:gradient_checker",
        "//machina/python/ops:nn_grad",
        "//machina/python/ops:nn_ops",
        "//machina/python/ops:random_ops",
        "//machina/python/ops:variable_scope",
        "//machina/python/ops:variables",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "conv3d_backprop_filter_v2_grad_test",
    size = "small",
    srcs = ["conv3d_backprop_filter_v2_grad_test.py"],
    deps = [
        "//machina/python/framework:constant_op",
        "//machina/python/framework:errors",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:gradient_checker",
        "//machina/python/ops:nn_grad",
        "//machina/python/ops:nn_ops",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "conv3d_transpose_test",
    size = "medium",
    srcs = ["conv3d_transpose_test.py"],
    deps = [
        "//machina/python/framework:constant_op",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:gradient_checker",
        "//machina/python/ops:nn_grad",
        "//machina/python/ops:nn_ops",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "conv_ops_3d_test",
    size = "medium",
    srcs = ["conv_ops_3d_test.py"],
    shard_count = 30,
    tags = [
        "no_cuda11",
        "no_mac_arm64",
    ],
    deps = [
        "//machina/python/eager:context",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/framework:tensor_shape",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:gradient_checker",
        "//machina/python/ops:gradients_impl",
        "//machina/python/ops:nn_grad",
        "//machina/python/ops:nn_ops",
        "//machina/python/ops:nn_ops_gen",
        "//machina/python/platform:client_testlib",
        "//machina/python/platform:tf_logging",
        "//machina/python/util:compat",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_strict_test(
    name = "conv_ops_test",
    size = "medium",
    srcs = ["conv_ops_test.py"],
    shard_count = 4,
    tags = [
        "no_mac_arm64",
        "optonly",  # times out
    ],
    deps = [
        "//machina/core:protos_all_py",
        "//machina/python/client:session",
        "//machina/python/eager:backprop",
        "//machina/python/eager:context",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/framework:ops",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:control_flow_ops",
        "//machina/python/ops:gradient_checker",
        "//machina/python/ops:gradients_impl",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:nn_grad",
        "//machina/python/ops:nn_impl",
        "//machina/python/ops:nn_ops",
        "//machina/python/ops:nn_ops_gen",
        "//machina/python/ops:random_ops",
        "//machina/python/ops:variables",
        "//machina/python/platform:client_testlib",
        "//machina/python/platform:tf_logging",
        "//machina/python/util:compat",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
    name = "ctc_decoder_ops_test",
    size = "small",
    srcs = ["ctc_decoder_ops_test.py"],
    deps = [
        "//machina/python/framework:errors",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops_stack",
        "//machina/python/ops:ctc_ops",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "ctc_loss_op_test",
    size = "medium",
    srcs = ["ctc_loss_op_test.py"],
    xla_enable_strict_auto_jit = False,  # b/148153828
    deps = [
        "//machina/python/eager:backprop",
        "//machina/python/eager:context",
        "//machina/python/eager:def_function",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:errors",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:random_seed",
        "//machina/python/framework:sparse_tensor",
        "//machina/python/framework:tensor_spec",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:ctc_ops",
        "//machina/python/ops:gradients_impl",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:random_ops",
        "//machina/python/ops:sparse_ops",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_library(
    name = "cudnn_deterministic_base",
    srcs = ["cudnn_deterministic_base.py"],
    deps = [
        "//machina/python/eager:backprop",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:nn_ops",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "cudnn_deterministic_ops_test",
    size = "small",
    srcs = ["cudnn_deterministic_ops_test.py"],
    tags = [
        "no_cuda_asan",  # TODO(b/171509035): re-enable.
        "no_rocm_pre_53",
    ],
    xla_enable_strict_auto_jit = True,
    deps = [
        ":cudnn_deterministic_base",
        "//machina/python/framework:config",
        "//machina/python/platform:client_testlib",
    ],
)

cuda_py_strict_test(
    name = "cudnn_d9m_test",
    size = "small",
    srcs = ["cudnn_d9m_test.py"],
    tags = [
        "no_cuda_asan",  # TODO(b/171509035): re-enable.
        "no_rocm",  #This is test is specific to CUDA and enables determinism through a CUDA specific env var.
    ],
    deps = [
        ":cudnn_deterministic_base",
        "//machina/python/platform:client_testlib",
    ],
)

py_strict_library(
    name = "depthwise_conv_op_base",
    srcs = ["depthwise_conv_op_base.py"],
    deps = [
        "//machina/python/framework:constant_op",
        "//machina/python/framework:errors",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:gradient_checker",
        "//machina/python/ops:nn_grad",
        "//machina/python/ops:nn_impl",
        "//machina/python/ops:nn_ops",
        "//machina/python/platform:client_testlib",
        "//machina/python/platform:tf_logging",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "depthwise_conv_op_d9m_test",
    size = "medium",  # http://b/30603882
    timeout = "long",
    srcs = ["depthwise_conv_op_d9m_test.py"],
    shard_count = 8,
    deps = [
        ":depthwise_conv_op_base",
        "//machina/python/eager:backprop",
        "//machina/python/framework:config",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:random_seed",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:nn_grad",
        "//machina/python/ops:nn_impl",
        "//machina/python/ops:random_ops",
        "//machina/python/platform:client_testlib",
    ],
)

cuda_py_strict_test(
    name = "depthwise_conv_op_test",
    size = "medium",  # http://b/30603882
    timeout = "long",
    srcs = ["depthwise_conv_op_test.py"],
    shard_count = 8,
    deps = [
        ":depthwise_conv_op_base",
        "//machina/python/platform:client_testlib",
    ],
)

cuda_py_strict_test(
    name = "embedding_ops_test",
    size = "medium",
    srcs = ["embedding_ops_test.py"],
    shard_count = 20,
    tags = [
        "no_cuda_asan",  # Size limit: b/192505612
    ],
    xla_tags = [
        "no_cuda_asan",  # Size limit: b/192505612
    ],
    deps = [
        "//machina/python/compat",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:ops",
        "//machina/python/framework:sparse_tensor",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:array_ops_stack",
        "//machina/python/ops:data_flow_ops",
        "//machina/python/ops:embedding_ops",
        "//machina/python/ops:gradient_checker",
        "//machina/python/ops:gradients",
        "//machina/python/ops:init_ops",
        "//machina/python/ops:linalg_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:partitioned_variables",
        "//machina/python/ops:resource_variable_ops",
        "//machina/python/ops:sort_ops",
        "//machina/python/ops:state_ops",
        "//machina/python/ops:variable_scope",
        "//machina/python/ops:variables",
        "//machina/python/ops/ragged:ragged_factory_ops",
        "//machina/python/ops/ragged:ragged_tensor",
        "//machina/python/platform:client_testlib",
        "//machina/python/platform:tf_logging",
        "//machina/python/util:compat",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
    name = "fractional_avg_pool_op_test",
    size = "small",
    srcs = ["fractional_avg_pool_op_test.py"],
    shard_count = 5,
    deps = [
        "//machina/python/framework:constant_op",
        "//machina/python/framework:errors",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:gradient_checker",
        "//machina/python/ops:nn_grad",
        "//machina/python/ops:nn_ops",
        "//machina/python/ops:nn_ops_gen",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

tf_py_strict_test(
    name = "fractional_max_pool_op_test",
    size = "small",
    srcs = ["fractional_max_pool_op_test.py"],
    shard_count = 5,
    deps = [
        "//machina/python/framework:constant_op",
        "//machina/python/framework:errors",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:gradient_checker",
        "//machina/python/ops:nn_grad",
        "//machina/python/ops:nn_ops",
        "//machina/python/ops:nn_ops_gen",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

tf_py_strict_test(
    name = "losses_test",
    size = "medium",
    srcs = ["losses_test.py"],
    deps = [
        "//machina/python/framework:constant_op",
        "//machina/python/framework:errors",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:random_seed",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:init_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:random_ops",
        "//machina/python/ops:variable_scope",
        "//machina/python/ops:variables",
        "//machina/python/ops/losses",
        "//machina/python/ops/losses:util",
        "//machina/python/platform:client_testlib",
        "//machina/python/training:momentum",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "lrn_op_test",
    size = "medium",
    srcs = ["lrn_op_test.py"],
    deps = [
        "//machina/python/framework:constant_op",
        "//machina/python/framework:errors",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:gradient_checker",
        "//machina/python/ops:gradients_impl",
        "//machina/python/ops:nn",
        "//machina/python/ops:nn_grad",
        "//machina/python/ops:random_ops",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "morphological_ops_test",
    size = "small",
    srcs = ["morphological_ops_test.py"],
    xla_tags = [
        "no_cuda_asan",  # times out
    ],
    deps = [
        "//machina/python/framework:config",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:errors",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:gradient_checker_v2",
        "//machina/python/ops:nn_grad",
        "//machina/python/ops:nn_ops",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_strict_test(
    name = "nth_element_op_test",
    size = "small",
    srcs = ["nth_element_op_test.py"],
    deps = [
        "//machina/python/eager:backprop",
        "//machina/python/eager:context",
        "//machina/python/framework:errors",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:gradients_impl",
        "//machina/python/ops:nn_grad",
        "//machina/python/ops:nn_ops",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "pool_test",
    size = "medium",
    srcs = ["pool_test.py"],
    xla_tags = [
        "no_cuda_asan",  # times out
    ],
    deps = [
        "//machina/python/framework:constant_op",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:gradient_checker",
        "//machina/python/ops:nn_grad",
        "//machina/python/ops:nn_ops",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "pooling_ops_3d_test",
    size = "medium",
    srcs = ["pooling_ops_3d_test.py"],
    deps = [
        "//machina/python/eager:backprop",
        "//machina/python/eager:context",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:gradient_checker_v2",
        "//machina/python/ops:nn_grad",
        "//machina/python/ops:nn_ops",
        "//machina/python/ops:nn_ops_gen",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "pooling_ops_test",
    size = "medium",
    srcs = ["pooling_ops_test.py"],
    shard_count = 10,
    # Some operations in this test can only be checked on sm61+.
    tags = ["prefer-sm70"],
    deps = [
        "//machina/python/eager:context",
        "//machina/python/framework:config",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:errors",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:array_ops_gen",
        "//machina/python/ops:gradient_checker",
        "//machina/python/ops:gradients_impl",
        "//machina/python/ops:nn_grad",
        "//machina/python/ops:nn_ops",
        "//machina/python/ops:nn_ops_gen",
        "//machina/python/ops:variables",
        "//machina/python/platform:client_testlib",
        "//machina/python/platform:tf_logging",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_strict_test(
    name = "relu_op_test",
    size = "small",
    srcs = ["relu_op_test.py"],
    xla_tags = [
        "no_cuda_asan",  # times out
    ],
    deps = [
        "//machina/python/eager:backprop",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:errors",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:gradient_checker_v2",
        "//machina/python/ops:nn_grad",
        "//machina/python/ops:nn_ops",
        "//machina/python/ops:random_ops",
        "//machina/python/ops:variables",
        "//machina/python/platform:client_testlib",
        "//machina/python/training:gradient_descent",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "rnn_cell_test",
    size = "medium",
    srcs = ["rnn_cell_test.py"],
    shard_count = 15,
    tags = ["no_windows"],  # b/139739217
    deps = [
        "//machina/core:protos_all_py",
        "//machina/python/checkpoint",
        "//machina/python/eager:context",
        "//machina/python/eager:def_function",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/framework:ops",
        "//machina/python/framework:random_seed",
        "//machina/python/framework:tensor",
        "//machina/python/framework:tensor_shape",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:array_ops_stack",
        "//machina/python/ops:cond",
        "//machina/python/ops:gradients_impl",
        "//machina/python/ops:init_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:random_ops",
        "//machina/python/ops:rnn",
        "//machina/python/ops:rnn_cell",
        "//machina/python/ops:rnn_cell_impl",
        "//machina/python/ops:rnn_ops_gen",
        "//machina/python/ops:state_ops",
        "//machina/python/ops:tensor_array_ops",
        "//machina/python/ops:variable_scope",
        "//machina/python/ops:variable_v1",
        "//machina/python/ops:variables",
        "//machina/python/platform:client_testlib",
        "//machina/python/platform:tf_logging",
        "//machina/python/saved_model:load",
        "//machina/python/saved_model:save",
        "//machina/python/trackable:autotrackable",
        "//machina/python/util:nest",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_strict_test(
    name = "rnn_test",
    size = "medium",
    timeout = "long",
    srcs = ["rnn_test.py"],
    shard_count = 10,
    deps = [
        "//machina/core:protos_all_py",
        "//machina/python/client:session",
        "//machina/python/eager:context",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:array_ops_stack",
        "//machina/python/ops:control_flow_ops",
        "//machina/python/ops:data_flow_grad",
        "//machina/python/ops:gradients_impl",
        "//machina/python/ops:init_ops",
        "//machina/python/ops:nn_grad",
        "//machina/python/ops:rnn",
        "//machina/python/ops:rnn_cell_impl",
        "//machina/python/ops:sparse_grad",
        "//machina/python/ops:tensor_array_grad",
        "//machina/python/ops:tensor_array_ops",
        "//machina/python/ops:variables",
        "//machina/python/platform:client_testlib",
        "//machina/python/training:saver",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "softmax_op_test",
    size = "medium",
    srcs = ["softmax_op_test.py"],
    deps = [
        "//machina/python/framework:constant_op",
        "//machina/python/framework:errors",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:nn_ops",
        "//machina/python/platform:client_testlib",
        "//machina/python/platform:tf_logging",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "softplus_op_test",
    size = "small",
    srcs = ["softplus_op_test.py"],
    deps = [
        "//machina/python/framework:constant_op",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:gradient_checker",
        "//machina/python/ops:gradients_impl",
        "//machina/python/ops:nn_grad",
        "//machina/python/ops:nn_ops",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "softsign_op_test",
    size = "small",
    srcs = ["softsign_op_test.py"],
    deps = [
        "//machina/python/framework:constant_op",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:gradient_checker",
        "//machina/python/ops:nn_grad",
        "//machina/python/ops:nn_ops",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "xent_op_d9m_test",
    size = "medium",
    timeout = "long",
    srcs = ["xent_op_d9m_test.py"],
    tags = [
        "no_pip",  # TODO(b/283889547) Flaky on kokoro
        "no_windows",  # Flaky on Windows CPU: https://github.com/machina/machina/issues/55827
        "nomac",  # TODO(b/235277289) Flaky on OSX
    ],
    xla_enable_strict_auto_jit = False,
    deps = [
        ":xent_op_test_base",
        "//machina/python/eager:backprop",
        "//machina/python/framework:config",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:nn_grad",
        "//machina/python/ops:nn_ops",
        "//machina/python/ops:nn_ops_gen",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "xent_op_test",
    size = "small",
    srcs = ["xent_op_test.py"],
    deps = [
        ":xent_op_test_base",
        "//machina/python/client:session",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:nn_ops",
        "//machina/python/ops:nn_ops_gen",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "xent_op_test_base",
    srcs = ["xent_op_test_base.py"],
    deps = [
        "//machina/python/eager:backprop",
        "//machina/python/framework:config",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:gradient_checker",
        "//machina/python/ops:gradients_impl",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:nn_grad",
        "//machina/python/ops:nn_ops",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

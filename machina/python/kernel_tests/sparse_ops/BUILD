# Tests of TensorFlow sparse ops written using the Python API.

load("//machina:strict.default.bzl", "py_strict_library")
load("//machina:machina.default.bzl", "cuda_py_strict_test", "tf_py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//machina:internal"],
    licenses = ["notice"],
)

tf_py_strict_test(
    name = "sparse_add_op_test",
    size = "small",
    srcs = ["sparse_add_op_test.py"],
    deps = [
        "//machina/python/client:session",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:errors",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:sparse_tensor",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:gradient_checker",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:sparse_grad",
        "//machina/python/ops:sparse_ops",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

tf_py_strict_test(
    name = "sparse_concat_op_test",
    size = "small",
    srcs = ["sparse_concat_op_test.py"],
    deps = [
        "//machina/python/framework:constant_op",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:sparse_tensor",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:sparse_ops",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

tf_py_strict_test(
    name = "sparse_conditional_accumulator_test",
    size = "small",
    srcs = ["sparse_conditional_accumulator_test.py"],
    deps = [
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/framework:indexed_slices",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor",
        "//machina/python/framework:tensor_shape",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:data_flow_ops",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

tf_py_strict_test(
    name = "sparse_cross_op_test",
    size = "small",
    srcs = ["sparse_cross_op_test.py"],
    tags = ["no_windows"],
    deps = [
        "//machina/python/client:session",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:errors",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:sparse_tensor",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:sparse_ops",
        "//machina/python/ops:sparse_ops_gen",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "sparse_matmul_op_test",
    size = "medium",
    srcs = ["sparse_matmul_op_test.py"],
    tags = ["no_windows"],
    deps = [
        "//machina/python/framework:constant_op",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:gradient_checker",
        "//machina/python/ops:math_ops",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "sparse_ops_test",
    size = "medium",
    srcs = ["sparse_ops_test.py"],
    shard_count = 5,
    tags = [
        "optonly",  # b/77589990
    ],
    deps = [
        "//machina/python/eager:def_function",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:errors",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:sparse_tensor",
        "//machina/python/framework:tensor_spec",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:gradient_checker",
        "//machina/python/ops:nn_ops",
        "//machina/python/ops:sparse_grad",
        "//machina/python/ops:sparse_ops",
        "//machina/python/ops:sparse_ops_gen",
        "//machina/python/ops:variables",
        "//machina/python/platform:client_testlib",
        "//machina/python/platform:test",
        "//third_party/py/numpy",
    ],
)

tf_py_strict_test(
    name = "sparse_reorder_op_test",
    size = "small",
    srcs = ["sparse_reorder_op_test.py"],
    deps = [
        "//machina/python/eager:def_function",
        "//machina/python/framework:errors",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:sparse_tensor",
        "//machina/python/framework:tensor_spec",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:gradient_checker",
        "//machina/python/ops:sparse_grad",
        "//machina/python/ops:sparse_ops",
        "//machina/python/ops:sparse_ops_gen",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
    name = "sparse_reshape_op_test",
    size = "small",
    srcs = ["sparse_reshape_op_test.py"],
    deps = [
        "//machina/python/framework:constant_op",
        "//machina/python/framework:errors",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:sparse_tensor",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:sparse_ops",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
    name = "sparse_serialization_ops_test",
    size = "small",
    srcs = ["sparse_serialization_ops_test.py"],
    deps = [
        "//machina/python/eager:def_function",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:sparse_tensor",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:array_ops_stack",
        "//machina/python/ops:resource_variable_ops_gen",
        "//machina/python/ops:sparse_ops",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

tf_py_strict_test(
    name = "sparse_slice_op_test",
    size = "small",
    srcs = ["sparse_slice_op_test.py"],
    deps = [
        "//machina/python/framework:errors",
        "//machina/python/framework:sparse_tensor",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:gradient_checker",
        "//machina/python/ops:sparse_grad",
        "//machina/python/ops:sparse_ops",
        "//machina/python/ops:sparse_ops_gen",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "sparse_split_op_test",
    size = "small",
    srcs = ["sparse_split_op_test.py"],
    deps = [
        "//machina/python/framework:errors",
        "//machina/python/framework:sparse_tensor",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:sparse_ops",
        "//machina/python/ops:sparse_ops_gen",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "sparse_tensor_dense_matmul_grad_test",
    size = "small",
    srcs = ["sparse_tensor_dense_matmul_grad_test.py"],
    xla_tags = [
        "no_cuda_asan",  # b/182392418 times out
    ],
    deps = [
        "//machina/python/framework:constant_op",
        "//machina/python/framework:sparse_tensor",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:gradient_checker",
        "//machina/python/ops:sparse_grad",
        "//machina/python/ops:sparse_ops",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    # Test name is shortened to workaround Windows Bazel's 259 char path limit.
    name = "sparse_tensor_dense_matmul_op_d9m_test",
    size = "medium",
    srcs = ["sparse_tensor_dense_matmul_op_d9m_test.py"],
    deps = [
        "//machina/python/framework:config",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:errors",
        "//machina/python/framework:sparse_tensor",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:sparse_ops",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "sparse_tensor_dense_matmul_op_test",
    size = "medium",
    srcs = ["sparse_tensor_dense_matmul_op_test.py"],
    deps = [
        "//machina/core:protos_all_py",
        "//machina/python/client:session",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:sparse_tensor",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:sparse_ops",
        "//machina/python/ops:while_loop",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl:app",
    ],
)

tf_py_strict_test(
    name = "sparse_tensors_map_ops_test",
    size = "small",
    srcs = ["sparse_tensors_map_ops_test.py"],
    deps = [
        "//machina/python/client:session",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:sparse_tensor",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:array_ops_stack",
        "//machina/python/ops:sparse_ops",
        "//machina/python/ops:variables",
        "//machina/python/platform:benchmark",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "sparse_to_dense_op_py_test",
    size = "small",
    srcs = ["sparse_to_dense_op_py_test.py"],
    deps = [
        "//machina/python/framework:constant_op",
        "//machina/python/framework:errors",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:sparse_ops",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_strict_test(
    name = "sparse_xent_op_d9m_test",
    size = "medium",
    srcs = ["sparse_xent_op_d9m_test.py"],
    tags = ["no_windows"],  # Fails as SegmentSum is nondeterministic on Windows
    xla_enable_strict_auto_jit = False,
    deps = [
        ":sparse_xent_op_test_base",
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
    name = "sparse_xent_op_test",
    size = "small",
    srcs = ["sparse_xent_op_test.py"],
    deps = [
        ":sparse_xent_op_test_base",
        "//machina/core:protos_all_py",
        "//machina/python/client:session",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:array_ops_stack",
        "//machina/python/ops:gradients_impl",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:nn_grad",
        "//machina/python/ops:sparse_ops",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl:app",
    ],
)

py_strict_library(
    name = "sparse_xent_op_test_base",
    srcs = ["sparse_xent_op_test_base.py"],
    deps = [
        "//machina/python/eager:backprop",
        "//machina/python/eager:context",
        "//machina/python/framework:config",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:gradient_checker_v2",
        "//machina/python/ops:nn_grad",
        "//machina/python/ops:nn_ops",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

tf_py_strict_test(
    name = "sparsemask_op_test",
    size = "small",
    srcs = ["sparsemask_op_test.py"],
    deps = [
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:indexed_slices",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

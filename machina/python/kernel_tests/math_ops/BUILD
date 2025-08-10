# Tests of TensorFlow math kernels written using the Python API.

load("//machina:machina.default.bzl", "cuda_py_strict_test", "tf_py_strict_test")
load(
    "//machina/tools/test:performance.bzl",
    "cuda_py_benchmark_test",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    licenses = ["notice"],
)

cuda_py_strict_test(
    name = "aggregate_ops_test",
    size = "small",
    srcs = ["aggregate_ops_test.py"],
    xla_tags = [
        "no_cuda_asan",  # times out
    ],
    deps = [
        "//machina/core:protos_all_py",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:string_ops",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "approx_topk_test",
    srcs = ["approx_topk_test.py"],
    tags = ["no_oss"],
    xla_enable_strict_auto_jit = True,
    xla_enabled = True,
    deps = [
        "//machina/python/eager:backprop",
        "//machina/python/eager:def_function",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:nn_ops",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_strict_test(
    name = "argmax_op_test",
    size = "small",
    srcs = ["argmax_op_test.py"],
    tags = [
        "no_cuda_asan",  # times out
    ],
    xla_tags = [
        "no_cuda_asan",  # times out
    ],
    deps = [
        "//machina/python/framework:dtypes",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_strict_test(
    name = "banded_triangular_solve_op_test",
    size = "small",
    srcs = ["banded_triangular_solve_op_test.py"],
    deps = [
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:linalg_ops",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "basic_gpu_test",
    size = "small",
    srcs = ["basic_gpu_test.py"],
    grpc_enabled = True,
    deps = [
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:random_seed",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:array_ops_gen",
        "//machina/python/ops:gradient_checker_v2",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:random_ops",
        "//machina/python/ops:variables",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "batch_matmul_op_test",
    size = "medium",
    srcs = ["batch_matmul_op_test.py"],
    shard_count = 20,
    tags = ["no_mac_arm64"],
    deps = [
        "//machina/python:tf2",
        "//machina/python/client:session",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:gradient_checker_v2",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:variables",
        "//machina/python/platform:benchmark",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "bincount_op_test",
    size = "small",
    srcs = ["bincount_op_test.py"],
    tags = ["no_windows_gpu"],
    deps = [
        "//machina/python/framework:constant_op",
        "//machina/python/framework:errors",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:bincount_ops",
        "//machina/python/ops:math_ops_gen",
        "//machina/python/ops:random_ops",
        "//machina/python/ops:sparse_ops",
        "//machina/python/ops/ragged:ragged_factory_ops",
        "//machina/python/ops/ragged:ragged_tensor",
        "//machina/python/platform:test",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_strict_test(
    name = "bucketize_op_test",
    size = "medium",
    srcs = ["bucketize_op_test.py"],
    deps = [
        "//machina/python/framework:constant_op",
        "//machina/python/framework:errors",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

tf_py_strict_test(
    name = "clip_ops_test",
    size = "small",
    srcs = ["clip_ops_test.py"],
    tags = [
        "no_gpu",  # b/127001953
        "no_windows",
    ],
    deps = [
        "//machina/python/eager:backprop",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:indexed_slices",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:clip_ops",
        "//machina/python/ops:gradients_impl",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:math_ops_gen",
        "//machina/python/platform:client_testlib",
    ],
)

tf_py_strict_test(
    name = "confusion_matrix_test",
    size = "small",
    srcs = ["confusion_matrix_test.py"],
    deps = [
        "//machina/python/framework:errors",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:confusion_matrix",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:random_ops",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "cross_grad_test",
    size = "small",
    srcs = ["cross_grad_test.py"],
    deps = [
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:gradient_checker",
        "//machina/python/ops:math_ops",
        "//machina/python/platform:client_testlib",
    ],
)

cuda_py_strict_test(
    name = "cumulative_logsumexp_test",
    size = "medium",
    srcs = ["cumulative_logsumexp_test.py"],
    deps = [
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:gradient_checker_v2",
        "//machina/python/ops:map_fn",
        "//machina/python/ops:math_ops",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "cwise_ops_test",
    size = "medium",
    srcs = ["cwise_ops_test.py"],
    shard_count = 50,
    tags = [
        "no_cuda_asan",  # b/179032113
    ],
    deps = [
        "//machina/python/eager:context",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:errors",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:sparse_tensor",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:gradient_checker",
        "//machina/python/ops:gradients_impl",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:nn_grad",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "cwise_ops_binary_test",
    size = "medium",
    srcs = ["cwise_ops_binary_test.py"],
    shard_count = 50,
    # b/140155647: Error just outside of tolerance
    xla_enable_strict_auto_jit = False,
    deps = [
        "//machina/python/framework:constant_op",
        "//machina/python/framework:errors",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:sparse_tensor",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:gradient_checker",
        "//machina/python/ops:gradients_impl",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:nn_grad",
        "//machina/python/ops:variables",
        "//machina/python/platform:client_testlib",
        "//machina/python/platform:tf_logging",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "cwise_ops_unary_test",
    size = "medium",
    srcs = ["cwise_ops_unary_test.py"],
    shard_count = 10,
    tags = [
        "no_windows",  # TODO(b/207048097): re-enable
    ],
    # b/140155706: nans in result
    xla_enable_strict_auto_jit = False,
    deps = [
        "//machina/python/eager:backprop",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:sparse_tensor",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:gradient_checker",
        "//machina/python/ops:gradient_checker_v2",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:math_ops_gen",
        "//machina/python/ops:nn_grad",
        "//machina/python/ops:special_math_ops",
        "//machina/python/platform:client_testlib",
        "//machina/python/platform:tf_logging",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "division_future_test",
    size = "medium",
    srcs = ["division_future_test.py"],
    tags = ["manual"],
    deps = [
        "//machina/python/framework:constant_op",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "division_past_test",
    size = "medium",
    srcs = ["division_past_test.py"],
    tags = [
        "manual",
        "no_windows",
    ],
    deps = [
        "//machina/python/framework:constant_op",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "in_topk_op_test",
    size = "small",
    srcs = ["in_topk_op_test.py"],
    deps = [
        "//machina/python/framework:constant_op",
        "//machina/python/ops:nn_ops",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "matmul_op_test",
    size = "medium",
    srcs = ["matmul_op_test.py"],
    shard_count = 20,
    tags = [
        "no_mac_arm64",
    ],
    deps = [
        "//machina/python:tf2",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:gradient_checker_v2",
        "//machina/python/ops:math_ops",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

cuda_py_benchmark_test(
    name = "reduce_benchmark_test",
    srcs = ["reduce_benchmark_test.py"],
    deps = [
        "//machina/core:protos_all_py",
        "//machina/python/client:session",
        "//machina/python/eager:backprop",
        "//machina/python/eager:context",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:ops",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:gradients_impl",
        "//machina/python/ops:math_ops",
        "//machina/python/platform:client_testlib",
    ],
)

cuda_py_strict_test(
    name = "reduction_ops_test_big",
    size = "medium",
    srcs = ["reduction_ops_test_big.py"],
    shard_count = 3,
    tags = [
        "manual",
        "no_gpu",
        "noguitar",
    ],
    deps = [
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "reduction_ops_test",
    size = "medium",
    srcs = ["reduction_ops_test.py"],
    shard_count = 6,
    tags = [
        "no_cuda_asan",  # times out
        "no_windows_gpu",
    ],
    deps = [
        "//machina/python/framework:constant_op",
        "//machina/python/framework:errors",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:gradient_checker",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:variables",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_strict_test(
    name = "segment_reduction_ops_d9m_test",
    size = "small",
    srcs = ["segment_reduction_ops_d9m_test.py"],
    tags = ["no_cuda_asan"],  # b/195247538
    xla_enable_strict_auto_jit = False,
    deps = [
        "//machina/python/eager:backprop",
        "//machina/python/framework:config",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:indexed_slices",
        "//machina/python/framework:ops",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:variables",
        "//machina/python/platform:client_testlib",
    ],
)

cuda_py_strict_test(
    name = "segment_reduction_ops_test",
    size = "medium",
    srcs = ["segment_reduction_ops_test.py"],
    shard_count = 10,
    # TODO (b/173835746): the test fails with XLA.
    xla_enable_strict_auto_jit = False,
    deps = [
        "//machina/python/client:session",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:errors",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:indexed_slices",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:gradient_checker",
        "//machina/python/ops:gradient_checker_v2",
        "//machina/python/ops:gradients",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:variables",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
    name = "sets_test",
    size = "medium",
    srcs = ["sets_test.py"],
    deps = [
        "//machina/python/framework:constant_op",
        "//machina/python/framework:errors",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:sparse_tensor",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:set_ops_gen",
        "//machina/python/ops:sets",
        "//machina/python/ops:sparse_ops",
        "//machina/python/platform:test",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_strict_test(
    name = "tensordot_op_test",
    size = "medium",
    srcs = ["tensordot_op_test.py"],
    shard_count = 20,
    tags = [
        "no_mac_arm64",
        "no_rocm",
    ],
    xla_enable_strict_auto_jit = False,  # b/161856380
    deps = [
        "//machina/python/eager:context",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:errors",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "topk_op_test",
    size = "medium",
    srcs = ["topk_op_test.py"],
    xla_tags = [
        "no_cuda_asan",  # times out
    ],
    deps = [
        "//machina/python/client:session",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:errors",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:gradients_impl",
        "//machina/python/ops:nn_grad",
        "//machina/python/ops:nn_ops",
        "//machina/python/ops:random_ops",
        "//machina/python/ops:resource_variable_ops",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "transpose_op_test",
    size = "medium",
    srcs = ["transpose_op_test.py"],
    shard_count = 10,
    tags = [
        "no_gpu",
        "no_oss",
        "notap",  # flaky timeout/segfault, b/136498892
        "optonly",  # times out
    ],
    deps = [
        "//machina/python/eager:def_function",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:errors",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:tensor_spec",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:gradient_checker_v2",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "zero_division_test",
    size = "medium",
    srcs = ["zero_division_test.py"],
    deps = [
        "//machina/python/framework:constant_op",
        "//machina/python/framework:errors",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:test_lib",
        "//machina/python/platform:client_testlib",
    ],
)

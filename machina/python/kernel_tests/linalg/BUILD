# Tests of TensorFlow linalg kernels written using the Python API.

load("//machina:machina.default.bzl", "cuda_py_strict_test", "tf_py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    licenses = ["notice"],
)

cuda_py_strict_test(
    name = "cholesky_op_test",
    size = "medium",
    srcs = ["cholesky_op_test.py"],
    shard_count = 5,
    deps = [
        "//machina/python/client:session",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:errors",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:control_flow_ops",
        "//machina/python/ops:gradient_checker_v2",
        "//machina/python/ops:linalg_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:stateless_random_ops",
        "//machina/python/ops:variables",
        "//machina/python/ops/linalg",
        "//machina/python/platform:benchmark",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "determinant_op_test",
    size = "medium",
    srcs = ["determinant_op_test.py"],
    deps = [
        "//machina/python/client:session",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:control_flow_ops",
        "//machina/python/ops:linalg_ops",
        "//machina/python/ops:linalg_ops_gen",
        "//machina/python/ops:random_ops",
        "//machina/python/ops:variables",
        "//machina/python/platform:benchmark",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

tf_py_strict_test(
    name = "eig_op_test",
    size = "medium",
    srcs = ["eig_op_test.py"],
    data = ["//machina/python/kernel_tests/linalg/testdata:self_adjoint_eig_op_test_files"],
    shard_count = 20,
    tags = [
        "no_windows",
        "nomsan",  # TODO(b/213581489): MSAN false positives from LAPACK use inside NumPy.
    ],
    deps = [
        "//machina/python/framework:constant_op",
        "//machina/python/framework:errors",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:gradient_checker_v2",
        "//machina/python/ops:linalg_ops",
        "//machina/python/ops:linalg_ops_gen",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:random_ops",
        "//machina/python/ops:sort_ops",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
    # b/127344411: xla_enable_strict_auto_jit = True,
)

cuda_py_strict_test(
    name = "einsum_op_test",
    size = "medium",
    srcs = ["einsum_op_test.py"],
    shard_count = 4,
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
        "//machina/python/ops:gradient_checker_v2",
        "//machina/python/ops:linalg_ops_gen",
        "//machina/python/ops:special_math_ops",
        "//machina/python/ops:variables",
        "//machina/python/platform:benchmark",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "linalg_grad_test",
    size = "medium",
    srcs = ["linalg_grad_test.py"],
    shard_count = 20,
    deps = [
        "//machina/python/framework:constant_op",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:gradient_checker_v2",
        "//machina/python/ops:gradients_impl",
        "//machina/python/ops:linalg_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops/linalg:linalg_impl",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "linalg_ops_test",
    size = "medium",
    srcs = ["linalg_ops_test.py"],
    shard_count = 8,
    tags = ["no_windows_gpu"],
    deps = [
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:linalg_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:random_ops",
        "//machina/python/ops/linalg",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_strict_test(
    name = "linear_operator_addition_test",
    size = "small",
    srcs = ["linear_operator_addition_test.py"],
    deps = [
        "//machina/python/framework:test_lib",
        "//machina/python/ops:linalg_ops",
        "//machina/python/ops/linalg",
        "//machina/python/ops/linalg:linear_operator_addition",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "linear_operator_adjoint_test",
    size = "medium",
    srcs = ["linear_operator_adjoint_test.py"],
    shard_count = 5,
    tags = [
        "optonly",  # times out
    ],
    deps = [
        "//machina/python/framework:config",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:variables",
        "//machina/python/ops/linalg",
        "//machina/python/ops/linalg:linear_operator",
        "//machina/python/ops/linalg:linear_operator_test_util",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "linear_operator_block_diag_test",
    size = "medium",
    srcs = ["linear_operator_block_diag_test.py"],
    shard_count = 8,
    tags = ["optonly"],
    deps = [
        "//machina/python/eager:backprop",
        "//machina/python/eager:context",
        "//machina/python/framework:config",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:variables",
        "//machina/python/ops/linalg",
        "//machina/python/ops/linalg:linear_operator",
        "//machina/python/ops/linalg:linear_operator_block_diag",
        "//machina/python/ops/linalg:linear_operator_test_util",
        "//machina/python/ops/linalg:linear_operator_util",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "linear_operator_block_lower_triangular_test",
    size = "medium",
    srcs = ["linear_operator_block_lower_triangular_test.py"],
    shard_count = 8,
    tags = [
        "no_gpu",  # Seg fault. http://b/365525243
        "optonly",
    ],
    deps = [
        "//machina/python/eager:backprop",
        "//machina/python/eager:context",
        "//machina/python/framework:config",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:variables",
        "//machina/python/ops/linalg",
        "//machina/python/ops/linalg:linear_operator_block_lower_triangular",
        "//machina/python/ops/linalg:linear_operator_test_util",
        "//machina/python/ops/linalg:linear_operator_util",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "linear_operator_composition_test",
    size = "medium",
    srcs = ["linear_operator_composition_test.py"],
    shard_count = 5,
    tags = [
        "optonly",  # times out
    ],
    deps = [
        "//machina/python/framework:config",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops/linalg",
        "//machina/python/ops/linalg:linear_operator_test_util",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "linear_operator_circulant_test",
    size = "medium",
    srcs = ["linear_operator_circulant_test.py"],
    shard_count = 50,
    tags = [
        "no_cuda11",  # TODO(b/197522782): reenable test after fixing.
        "optonly",  # times out, b/79171797
    ],
    deps = [
        "//machina/python/framework:config",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:array_ops_stack",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:variables",
        "//machina/python/ops/linalg",
        "//machina/python/ops/linalg:linear_operator_circulant",
        "//machina/python/ops/linalg:linear_operator_test_util",
        "//machina/python/ops/signal:fft_ops",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_strict_test(
    name = "linear_operator_diag_test",
    size = "medium",
    srcs = ["linear_operator_diag_test.py"],
    shard_count = 5,
    tags = ["optonly"],
    deps = [
        "//machina/python/framework:config",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:linalg_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:random_ops",
        "//machina/python/ops:variables",
        "//machina/python/ops/linalg",
        "//machina/python/ops/linalg:linear_operator_test_util",
        "//machina/python/platform:client_testlib",
    ],
)

cuda_py_strict_test(
    name = "linear_operator_full_matrix_test",
    size = "medium",
    srcs = ["linear_operator_full_matrix_test.py"],
    shard_count = 5,
    tags = ["optonly"],
    deps = [
        "//machina/python/framework:config",
        "//machina/python/framework:errors",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:variables",
        "//machina/python/ops/linalg",
        "//machina/python/ops/linalg:linear_operator_test_util",
        "//machina/python/platform:client_testlib",
    ],
)

cuda_py_strict_test(
    name = "linear_operator_householder_test",
    size = "medium",
    srcs = ["linear_operator_householder_test.py"],
    shard_count = 5,
    tags = ["optonly"],
    deps = [
        "//machina/python/framework:config",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:linalg_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:variables",
        "//machina/python/ops/linalg",
        "//machina/python/ops/linalg:linear_operator_householder",
        "//machina/python/ops/linalg:linear_operator_test_util",
        "//machina/python/platform:client_testlib",
    ],
)

cuda_py_strict_test(
    name = "linear_operator_identity_test",
    size = "medium",
    srcs = ["linear_operator_identity_test.py"],
    shard_count = 5,
    tags = ["optonly"],
    deps = [
        "//machina/python/framework:config",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:linalg_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:random_ops",
        "//machina/python/ops:variables",
        "//machina/python/ops/linalg",
        "//machina/python/ops/linalg:linear_operator_test_util",
        "//machina/python/platform:client_testlib",
    ],
)

cuda_py_strict_test(
    name = "linear_operator_inversion_test",
    size = "medium",
    srcs = ["linear_operator_inversion_test.py"],
    shard_count = 5,
    tags = [
        "optonly",  # times out
    ],
    deps = [
        "//machina/python/framework:config",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:variables",
        "//machina/python/ops/linalg",
        "//machina/python/ops/linalg:linear_operator",
        "//machina/python/ops/linalg:linear_operator_test_util",
        "//machina/python/platform:client_testlib",
    ],
)

cuda_py_strict_test(
    name = "linear_operator_kronecker_test",
    size = "medium",
    srcs = ["linear_operator_kronecker_test.py"],
    shard_count = 10,
    tags = [
        "no_oss",  # TODO(b/282984925)
        "optonly",
    ],
    xla_enable_strict_auto_jit = False,
    deps = [
        "//machina/python/eager:backprop",
        "//machina/python/framework:config",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:variables",
        "//machina/python/ops/linalg",
        "//machina/python/ops/linalg:linear_operator",
        "//machina/python/ops/linalg:linear_operator_kronecker",
        "//machina/python/ops/linalg:linear_operator_test_util",
        "//machina/python/ops/linalg:linear_operator_util",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "linear_operator_low_rank_update_test",
    size = "medium",
    srcs = ["linear_operator_low_rank_update_test.py"],
    shard_count = 15,
    tags = ["optonly"],
    deps = [
        "//machina/python/framework:config",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:variables",
        "//machina/python/ops/linalg",
        "//machina/python/ops/linalg:linear_operator_test_util",
        "//machina/python/platform:client_testlib",
    ],
)

cuda_py_strict_test(
    name = "linear_operator_lower_triangular_test",
    size = "medium",
    srcs = ["linear_operator_lower_triangular_test.py"],
    shard_count = 4,
    tags = ["optonly"],
    deps = [
        "//machina/python/framework:config",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:variables",
        "//machina/python/ops/linalg",
        "//machina/python/ops/linalg:linear_operator_test_util",
        "//machina/python/platform:client_testlib",
    ],
)

cuda_py_strict_test(
    name = "linear_operator_permutation_test",
    size = "medium",
    srcs = ["linear_operator_permutation_test.py"],
    shard_count = 5,
    tags = ["optonly"],
    xla_enable_strict_auto_jit = True,
    deps = [
        "//machina/python/framework:config",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:linalg_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:random_ops",
        "//machina/python/ops/linalg",
        "//machina/python/ops/linalg:linear_operator_permutation",
        "//machina/python/ops/linalg:linear_operator_test_util",
        "//machina/python/platform:client_testlib",
    ],
)

cuda_py_strict_test(
    name = "linear_operator_test",
    size = "small",
    srcs = ["linear_operator_test.py"],
    deps = [
        "//machina/python/eager:context",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:linalg_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops/linalg",
        "//machina/python/ops/parallel_for:control_flow_ops",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "linear_operator_toeplitz_test",
    size = "medium",
    srcs = ["linear_operator_toeplitz_test.py"],
    shard_count = 5,
    tags = [
        "no_cuda_on_cpu_tap",  # flaky, b/135701551
        "no_gpu",  # flaky, b/135701551
        "optonly",  # times out, b/79171797
    ],
    deps = [
        "//machina/python/eager:backprop",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:variables",
        "//machina/python/ops/linalg",
        "//machina/python/ops/linalg:linear_operator_test_util",
        "//machina/python/ops/linalg:linear_operator_toeplitz",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "linear_operator_tridiag_test",
    size = "medium",
    srcs = ["linear_operator_tridiag_test.py"],
    shard_count = 10,
    tags = [
        "no_windows_gpu",
        "optonly",
    ],
    # TODO(b/313470344): XLA temporarily disabled due to empty shards on 3.12.
    xla_enable_strict_auto_jit = False,
    xla_enabled = False,
    deps = [
        "//machina/python/framework:config",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:array_ops_gen",
        "//machina/python/ops:array_ops_stack",
        "//machina/python/ops:manip_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:variables",
        "//machina/python/ops/linalg",
        "//machina/python/ops/linalg:linear_operator_test_util",
        "//machina/python/platform:client_testlib",
    ],
)

cuda_py_strict_test(
    name = "linear_operator_util_test",
    size = "medium",
    srcs = ["linear_operator_util_test.py"],
    shard_count = 5,
    tags = ["optonly"],
    deps = [
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor",
        "//machina/python/framework:tensor_shape",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:linalg_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops/linalg",
        "//machina/python/ops/linalg:linear_operator_util",
        "//machina/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_strict_test(
    name = "linear_operator_zeros_test",
    size = "medium",
    srcs = ["linear_operator_zeros_test.py"],
    shard_count = 5,
    tags = ["optonly"],  # Test is flaky without optimization.
    deps = [
        "//machina/python/framework:dtypes",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:variables",
        "//machina/python/ops/linalg",
        "//machina/python/ops/linalg:linear_operator_test_util",
        "//machina/python/platform:client_testlib",
    ],
)

cuda_py_strict_test(
    name = "lu_op_test",
    size = "small",
    srcs = ["lu_op_test.py"],
    xla_tags = [
        "no_cuda_asan",  # times out
    ],
    deps = [
        "//machina/python/client:session",
        "//machina/python/framework:errors",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:array_ops_stack",
        "//machina/python/ops:control_flow_ops",
        "//machina/python/ops:linalg_ops",
        "//machina/python/ops:map_fn",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:stateless_random_ops",
        "//machina/python/ops:variables",
        "//machina/python/platform:benchmark",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

tf_py_strict_test(
    name = "matrix_exponential_op_test",
    size = "medium",
    srcs = ["matrix_exponential_op_test.py"],
    shard_count = 16,
    tags = ["no_windows_gpu"],
    deps = [
        "//machina/python/client:session",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:control_flow_ops",
        "//machina/python/ops:random_ops",
        "//machina/python/ops:variables",
        "//machina/python/ops/linalg:linalg_impl",
        "//machina/python/platform:benchmark",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "matrix_inverse_op_test",
    size = "small",
    timeout = "moderate",
    srcs = ["matrix_inverse_op_test.py"],
    tags = ["optonly"],
    deps = [
        "//machina/python/client:session",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:control_flow_ops",
        "//machina/python/ops:linalg_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:random_ops",
        "//machina/python/ops:variables",
        "//machina/python/platform:benchmark",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

tf_py_strict_test(
    name = "matrix_logarithm_op_test",
    size = "medium",
    srcs = ["matrix_logarithm_op_test.py"],
    deps = [
        "//machina/python/client:session",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:errors",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:control_flow_ops",
        "//machina/python/ops:linalg_ops_gen",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:stateless_random_ops",
        "//machina/python/ops:variables",
        "//machina/python/ops/linalg:linalg_impl",
        "//machina/python/platform:benchmark",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "matrix_solve_ls_op_test",
    size = "medium",
    srcs = ["matrix_solve_ls_op_test.py"],
    deps = [
        "//machina/python/client:session",
        "//machina/python/eager:context",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:errors",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:control_flow_ops",
        "//machina/python/ops:linalg_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:variables",
        "//machina/python/platform:benchmark",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "matrix_solve_op_test",
    size = "medium",
    srcs = ["matrix_solve_op_test.py"],
    deps = [
        "//machina/python/client:session",
        "//machina/python/eager:context",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:errors",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:control_flow_ops",
        "//machina/python/ops:linalg_ops",
        "//machina/python/ops:stateless_random_ops",
        "//machina/python/ops:variables",
        "//machina/python/platform:benchmark",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "matrix_square_root_op_test",
    size = "medium",
    srcs = ["matrix_square_root_op_test.py"],
    deps = [
        "//machina/python/framework:constant_op",
        "//machina/python/framework:errors",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:linalg_ops_gen",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:stateless_random_ops",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "matrix_triangular_solve_op_test",
    size = "medium",
    srcs = ["matrix_triangular_solve_op_test.py"],
    shard_count = 3,
    deps = [
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:linalg_ops",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "normalize_op_test",
    size = "medium",
    srcs = ["normalize_op_test.py"],
    shard_count = 20,
    tags = ["no_windows_gpu"],
    # TODO(b/208263392): Re-enable. tf.Squeeze op after tf.Where op doesn't reshape.
    xla_enable_strict_auto_jit = False,
    deps = [
        "//machina/python/framework:test_lib",
        "//machina/python/ops:nn_impl",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "norm_op_test",
    size = "medium",
    srcs = ["norm_op_test.py"],
    shard_count = 20,
    tags = ["no_windows_gpu"],
    # TODO(b/208263392): Re-enable. tf.Squeeze op after tf.Where op doesn't reshape.
    xla_enable_strict_auto_jit = False,
    deps = [
        "//machina/python/framework:constant_op",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:linalg_ops",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "qr_op_test",
    size = "medium",
    srcs = ["qr_op_test.py"],
    shard_count = 20,
    deps = [
        "//machina/python/client:session",
        "//machina/python/eager:backprop",
        "//machina/python/eager:context",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:errors",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:control_flow_ops",
        "//machina/python/ops:gradient_checker_v2",
        "//machina/python/ops:linalg_ops",
        "//machina/python/ops:stateless_random_ops",
        "//machina/python/ops:variables",
        "//machina/python/platform:benchmark",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "self_adjoint_eig_op_test",
    size = "medium",
    srcs = ["self_adjoint_eig_op_test.py"],
    data = ["//machina/python/kernel_tests/linalg/testdata:self_adjoint_eig_op_test_files"],
    shard_count = 20,
    tags = [
        "no_windows",
    ],
    deps = [
        "//machina/python/framework:constant_op",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:gradient_checker_v2",
        "//machina/python/ops:linalg_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:random_ops",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
    # TODO(b/127344411): This test passes because XLA does not actually cluster
    # the self_adjoint_eig op.
)

cuda_py_strict_test(
    name = "slicing_test",
    size = "small",
    srcs = ["slicing_test.py"],
    deps = [
        "//machina/python/framework:constant_op",
        "//machina/python/framework:errors",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops/linalg",
        "//machina/python/ops/linalg:linear_operator_test_util",
        "//machina/python/ops/linalg:slicing",
        "//machina/python/platform:client_testlib",
    ],
)

cuda_py_strict_test(
    name = "svd_op_test",
    size = "medium",
    srcs = ["svd_op_test.py"],
    shard_count = 30,
    tags = [
        "no_oss",  # b/117185141.
    ],
    xla_tags = [
        "no_cuda_asan",  # times out
    ],
    deps = [
        "//machina/python/client:session",
        "//machina/python/eager:context",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:errors",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:control_flow_ops",
        "//machina/python/ops:gradient_checker",
        "//machina/python/ops:gradient_checker_v2",
        "//machina/python/ops:gradients_impl",
        "//machina/python/ops:linalg_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:stateless_random_ops",
        "//machina/python/ops:variables",
        "//machina/python/platform:benchmark",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
    # TODO(b/127344411): This test passes because XLA does not actually cluster
    # the svd op.
)

cuda_py_strict_test(
    name = "tridiagonal_matmul_op_test",
    size = "medium",
    srcs = ["tridiagonal_matmul_op_test.py"],
    shard_count = 5,
    deps = [
        "//machina/python/client:session",
        "//machina/python/eager:context",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:errors",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:array_ops_stack",
        "//machina/python/ops:control_flow_ops",
        "//machina/python/ops:gradient_checker_v2",
        "//machina/python/ops:linalg_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:variables",
        "//machina/python/ops/linalg:linalg_impl",
        "//machina/python/platform:benchmark",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "tridiagonal_solve_op_test",
    size = "medium",
    srcs = ["tridiagonal_solve_op_test.py"],
    shard_count = 12,
    tags = [
        "no_oss",  # TODO(b/142818120): Re-enable.
    ],
    deps = [
        "//machina/python/client:session",
        "//machina/python/eager:backprop",
        "//machina/python/eager:context",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:control_flow_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:variables",
        "//machina/python/ops/linalg:linalg_impl",
        "//machina/python/platform:benchmark",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

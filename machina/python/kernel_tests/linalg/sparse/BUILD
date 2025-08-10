# Tests of TensorFlow sparse linear algebra kernels using the Python API.

load("//machina:machina.default.bzl", "cuda_py_strict_test", "tf_py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//machina:internal"],
    licenses = ["notice"],
)

cuda_py_strict_test(
    name = "conjugate_gradient_test",
    size = "medium",
    srcs = ["conjugate_gradient_test.py"],
    deps = [
        "//machina/python/framework:constant_op",
        "//machina/python/ops:array_ops",
        "//machina/python/ops/linalg",
        "//machina/python/ops/linalg/sparse:conjugate_gradient",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_strict_test(
    name = "csr_sparse_matrix_test",
    size = "medium",
    srcs = ["csr_sparse_matrix_test.py"],
    main = "csr_sparse_matrix_test.py",
    tags = [
        "no_cuda_asan",  # TODO(b/190824595)
    ],
    deps = [
        "//machina/python/framework:dtypes",
        "//machina/python/framework:sparse_tensor",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:math_ops",
        "//machina/python/ops/linalg/sparse:sparse_csr_matrix_ops",
        "//machina/python/platform:client_testlib",
    ],
)

cuda_py_strict_test(
    name = "csr_sparse_matrix_ops_test",
    size = "medium",
    timeout = "long",
    srcs = ["csr_sparse_matrix_ops_test.py"],
    main = "csr_sparse_matrix_ops_test.py",
    shard_count = 10,
    tags = [
        "no_gpu",  # b/203655060 (cuda 11.5 specific)
    ],
    deps = [
        "//machina/core:protos_all_py",
        "//machina/python/client:session",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/framework:ops",
        "//machina/python/framework:random_seed",
        "//machina/python/framework:sparse_tensor",
        "//machina/python/framework:tensor_shape",
        "//machina/python/framework:tensor_util",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:array_ops_stack",
        "//machina/python/ops:linalg_ops",
        "//machina/python/ops:map_fn",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:nn_ops",
        "//machina/python/ops:random_ops",
        "//machina/python/ops:sparse_ops",
        "//machina/python/ops:state_ops",
        "//machina/python/ops:variable_scope",
        "//machina/python/ops/linalg/sparse:sparse_csr_matrix_ops",
        "//machina/python/platform:client_testlib",
        "//machina/python/platform:tf_logging",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_strict_test(
    name = "csr_sparse_matrix_grad_test",
    size = "medium",
    srcs = ["csr_sparse_matrix_grad_test.py"],
    main = "csr_sparse_matrix_grad_test.py",
    shard_count = 3,
    deps = [
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:gradients_impl",
        "//machina/python/ops:math_ops",
        "//machina/python/ops/linalg/sparse:sparse_csr_matrix_grad",
        "//machina/python/ops/linalg/sparse:sparse_csr_matrix_ops",
        "//machina/python/platform:client_testlib",
        "//machina/python/platform:tf_logging",
    ],
)

cuda_py_strict_test(
    name = "csr_sparse_matrix_dense_mat_mul_grad_test",
    size = "medium",
    srcs = ["csr_sparse_matrix_dense_mat_mul_grad_test.py"],
    main = "csr_sparse_matrix_dense_mat_mul_grad_test.py",
    shard_count = 50,
    tags = [
        "no_cuda_asan",  # TODO(b/190824595)
    ],
    deps = [
        "//machina/python/framework:ops",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:gradient_checker",
        "//machina/python/ops/linalg/sparse:sparse_csr_matrix_grad",
        "//machina/python/ops/linalg/sparse:sparse_csr_matrix_ops",
        "//machina/python/platform:client_testlib",
        "//machina/python/platform:tf_logging",
    ],
)

tf_py_strict_test(
    name = "csr_sparse_matrix_dense_mat_mul_onednn_grad_test",
    size = "medium",
    srcs = ["csr_sparse_matrix_dense_mat_mul_grad_test.py"],
    env = {
        "TF_ENABLE_ONEDNN_SPMM": "1",
    },
    main = "csr_sparse_matrix_dense_mat_mul_grad_test.py",
    shard_count = 50,
    tags = [
        "no_cuda_asan",  # TODO(b/190824595)
    ],
    deps = [
        "//machina/python/framework:ops",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:gradient_checker",
        "//machina/python/ops/linalg/sparse:sparse_csr_matrix_grad",
        "//machina/python/ops/linalg/sparse:sparse_csr_matrix_ops",
        "//machina/python/platform:client_testlib",
        "//machina/python/platform:tf_logging",
    ],
)

cuda_py_strict_test(
    name = "csr_sparse_matrix_sparse_mat_mul_grad_test",
    size = "medium",
    srcs = ["csr_sparse_matrix_sparse_mat_mul_grad_test.py"],
    main = "csr_sparse_matrix_sparse_mat_mul_grad_test.py",
    shard_count = 10,
    deps = [
        "//machina/python/framework:ops",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:gradient_checker",
        "//machina/python/ops:math_ops",
        "//machina/python/ops/linalg/sparse:sparse_csr_matrix_grad",
        "//machina/python/ops/linalg/sparse:sparse_csr_matrix_ops",
        "//machina/python/platform:client_testlib",
        "//machina/python/platform:tf_logging",
    ],
)

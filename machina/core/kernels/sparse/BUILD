# Description: Op kernels for sparse matrix operations.

load(
    "@local_config_rocm//rocm:build_defs.bzl",
    "if_rocm",
)
load(
    "//machina:machina.bzl",
    "if_cuda_or_rocm",
    "tf_cc_test",
)
load("//machina:machina.default.bzl", "tf_kernel_library")
load("//machina/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "sparse_matrix",
    srcs = ["sparse_matrix.cc"],
    hdrs = ["sparse_matrix.h"],
    deps = [
        "//machina/core:framework",
        "//machina/core/platform:errors",
        "@eigen_archive//:eigen3",
    ],
    alwayslink = 1,
)

tf_kernel_library(
    name = "kernels",
    srcs = [
        "add_op.cc",
        "conj_op.cc",
        "csr_sparse_matrix_to_dense_op.cc",
        "csr_sparse_matrix_to_sparse_tensor_op.cc",
        "dense_to_csr_sparse_matrix_op.cc",
        "kernels.cc",
        "mat_mul_op.cc",
        "mul_op.cc",
        "nnz_op.cc",
        "softmax_op.cc",
        "sparse_cholesky_op.cc",
        "sparse_mat_mul_op.cc",
        "sparse_matrix_components_op.cc",
        "sparse_ordering_amd_op.cc",
        "sparse_tensor_to_csr_sparse_matrix_op.cc",
        "transpose_op.cc",
        "zeros_op.cc",
    ],
    hdrs = [
        "kernels.h",
        "mat_mul_op.h",
        "transpose_op.h",
        "zeros_op.h",
    ],
    gpu_deps = [
        "//machina/core/kernels:gpu_device_array",
    ],
    gpu_srcs = [
        "zeros_op.h",
        "kernels.h",
        "kernels_gpu.cu.cc",
    ],
    deps = [
        ":sparse_matrix",
        "//machina/core:array_ops_op_lib",
        "//machina/core:bitwise_ops_op_lib",
        "//machina/core:framework",
        "//machina/core:framework_internal",
        "//machina/core:functional_ops_op_lib",
        "//machina/core:lib",
        "//machina/core:math_ops_op_lib",
        "//machina/core:nn_ops_op_lib",
        "//machina/core:no_op_op_lib",
        "//machina/core:sendrecv_ops_op_lib",
        "//machina/core:sparse_csr_matrix_ops_op_lib",
        "//machina/core:state_ops_op_lib",
        "//machina/core/framework:types_proto_cc",
        "//machina/core/kernels:concat_lib",
        "//machina/core/kernels:constant_op",
        "//machina/core/kernels:cwise_op",
        "//machina/core/kernels:dense_update_functor",
        "//machina/core/kernels:fill_functor",
        "//machina/core/kernels:gather_nd_op",
        "//machina/core/kernels:gpu_prim_hdrs",
        "//machina/core/kernels:scatter_nd_op",
        "//machina/core/kernels:slice_op",
        "//machina/core/kernels:sparse_utils",
        "//machina/core/kernels:transpose_functor",
        "@com_google_absl//absl/status",
        "@eigen_archive//:eigen3",
    ] + if_cuda_or_rocm([
        "//machina/core/util:cuda_solvers",
        "//machina/core/util:cuda_sparse",
    ]) + if_rocm([
        "//machina/core/util:rocm_solvers",
    ]),
    alwayslink = 1,
)

tf_cc_test(
    name = "kernels_test",
    size = "small",
    srcs = [
        "kernels_test.cc",
    ],
    deps = [
        ":kernels",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:test",
        "//machina/core:testlib",
        "//machina/core/framework:types_proto_cc",
        "@com_google_googletest//:gtest",
        "@eigen_archive//:eigen3",
        "@local_tsl//tsl/platform:status_matchers",
    ],
)

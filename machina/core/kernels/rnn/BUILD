# Description:
#   OpKernels for RNN ops.

load(
    "@local_config_rocm//rocm:build_defs.bzl",
    "if_rocm_is_configured",
)
load(
    "@local_xla//xla/tsl/platform/default:cuda_build_defs.bzl",
    "if_cuda_is_configured",
)
load(
    "//machina:machina.bzl",
    "tf_gpu_library",
)
load("//machina:machina.default.bzl", "tf_kernel_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//machina:internal"],
)

licenses(["notice"])

tf_gpu_library(
    name = "blas_gemm",
    srcs = [] + if_cuda_is_configured([
        "blas_gemm.cc",
    ]) + if_rocm_is_configured([
        "blas_gemm.cc",
    ]),
    hdrs = ["blas_gemm.h"],
    deps = [
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core/kernels:eigen_helpers",
        "//machina/core/kernels:numeric_options_utils",
        "//machina/core/platform:stream_executor",
        "@eigen_archive//:eigen3",
        "@local_xla//xla/tsl/framework/contraction:eigen_contraction_kernel",
    ],
)

tf_kernel_library(
    name = "lstm_ops",
    prefix = "lstm_ops",
    deps = [
        ":blas_gemm",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core/kernels:eigen_helpers",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@eigen_archive//:eigen3",
    ],
)

tf_kernel_library(
    name = "gru_ops",
    prefix = "gru_ops",
    deps = [
        ":blas_gemm",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core/kernels:eigen_helpers",
        "@eigen_archive//:eigen3",
    ],
)

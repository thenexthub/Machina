load("//machina:machina.default.bzl", "tf_kernel_library")

# Implementation of Special Functions kernels.

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

tf_kernel_library(
    name = "special_math_op",
    gpu_deps = ["//machina/core/kernels:cwise_lib"],
    prefix = "special_math_op",
    deps = [
        "//machina/core:core_cpu",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core:math_grad",
        "//machina/core/framework:bounds_check",
        "//machina/core/kernels:cwise_op",
        "//machina/core/kernels:fill_functor",
        "@eigen_archive//:eigen3",
    ],
)

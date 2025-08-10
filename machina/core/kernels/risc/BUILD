load("//machina:machina.default.bzl", "tf_kernel_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

tf_kernel_library(
    name = "risc",
    visibility = ["//visibility:public"],
    deps = [
        "//machina/core/kernels/risc/experimental",
    ],
)

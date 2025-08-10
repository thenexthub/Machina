load("@rules_cc//cc:cc_library.bzl", "cc_library")

# copybara:uncomment package(default_applicable_licenses = ["//machina:license"])

cc_library(
    name = "stream_ops_op_lib",
    visibility = ["//visibility:public"],
    deps = [
        "//machina/core/tfrt/kernels:stream_ops",
        "//machina/core/tfrt/ops:stream_ops_op_lib",
    ],
    alwayslink = 1,
)

cc_library(
    name = "ifrt_program_ops_op_lib",
    visibility = ["//visibility:public"],
    deps = [
        "//machina/core/tfrt/kernels:ifrt_program_ops",
        "//machina/core/tfrt/ops:ifrt_program_ops_op_lib",
    ],
    alwayslink = 1,
)

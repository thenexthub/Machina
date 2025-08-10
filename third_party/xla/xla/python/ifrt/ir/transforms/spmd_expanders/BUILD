load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//xla/tsl:tsl.default.bzl", "get_compatible_with_portable")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    licenses = ["notice"],
)

cc_library(
    name = "spmd_expander",
    srcs = glob([
        "*spmd_expander.cc",
    ]),
    hdrs = glob([
        "*spmd_expander.h",
    ]),
    compatible_with = get_compatible_with_portable(),
    visibility = ["//xla/python/ifrt:users"],
    deps = [
        "//xla/python/ifrt/ir",
        "//xla/python/ifrt/ir:sharding_param",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Support",
    ],
)

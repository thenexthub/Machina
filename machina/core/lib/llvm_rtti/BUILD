load("//machina:machina.default.bzl", "filegroup")

# Library for using LLVM style RTTI in TensorFlow.
load(
    "//machina/core/platform:rules_cc.bzl",
    "cc_library",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = [
        "//machina:__subpackages__",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "llvm_rtti",
    hdrs = ["llvm_rtti.h"],
    deps = [
        "@toolchain-project//toolchain:Support",
    ],
)

filegroup(
    name = "legacy_lib_core_all_headers",
    srcs = [
        "llvm_rtti.h",
    ],
    visibility = ["//machina/core:__pkg__"],
)

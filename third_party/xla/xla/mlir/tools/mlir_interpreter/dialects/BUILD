load("//xla:xla.default.bzl", "xla_cc_test")
load("//xla/tsl:tsl.bzl", "if_windows")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = [":friends"],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    includes = [
        "//xla:friends",
    ],
)

cc_library(
    name = "dialects",
    srcs = glob(
        [
            "*.cc",
        ],
        exclude = [
            "util.cc",
            "symbol_finder_linux.cc",
            "symbol_finder_windows.cc",
            "symbol_finder_test.cc",
        ],
    ),
    deps = [
        ":dialect_utils",
        ":symbol_finder",
        "//xla/mlir/tools/mlir_interpreter/framework",
        "//xla/mlir_hlo",
        "@com_google_absl//absl/status:statusor",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:AffineDialect",
        "@toolchain-project//mlir:AffineUtils",
        "@toolchain-project//mlir:ArithDialect",
        "@toolchain-project//mlir:BufferizationDialect",
        "@toolchain-project//mlir:ComplexDialect",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:LinalgDialect",
        "@toolchain-project//mlir:MemRefDialect",
        "@toolchain-project//mlir:SCFDialect",
        "@toolchain-project//mlir:Support",
        "@toolchain-project//mlir:TensorDialect",
        "@toolchain-project//mlir:VectorDialect",
    ],
    alwayslink = 1,
)

cc_library(
    name = "dialect_utils",
    srcs = [
        "util.cc",
    ],
    hdrs = [
        "comparators.h",
        "cwise_math.h",
        "util.h",
    ],
    deps = [
        "//xla/mlir/tools/mlir_interpreter/framework",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Support",
        "@toolchain-project//mlir:ViewLikeInterface",
    ],
)

cc_library(
    name = "symbol_finder_linux",
    srcs = [
        "symbol_finder.h",
        "symbol_finder_linux.cc",
    ],
    tags = ["manual"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "symbol_finder_windows",
    srcs = [
        "symbol_finder.h",
        "symbol_finder_windows.cc",
    ],
    tags = [
        "manual",
        "nobuilder",
    ],
    deps = [
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "symbol_finder",
    hdrs = ["symbol_finder.h"],
    deps = [
        "@com_google_absl//absl/status:statusor",
    ] + if_windows(
        [":symbol_finder_windows"],
        [":symbol_finder_linux"],
    ),
)

xla_cc_test(
    name = "symbol_finder_test",
    srcs = ["symbol_finder_test.cc"],
    deps = [
        ":symbol_finder",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:status_matchers",
        "@local_tsl//tsl/platform:test",
    ],
)

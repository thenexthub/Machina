# MLIR related utilities.

load(
    "//xla/tsl/platform:rules_cc.bzl",
    "cc_library",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "status_scoped_diagnostic_handler",
    srcs = ["status_scoped_diagnostic_handler.cc"],
    hdrs = [
        "status_scoped_diagnostic_handler.h",
    ],
    deps = [
        "//xla/tsl/platform:logging",
        "@com_google_absl//absl/status",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Support",
    ],
)

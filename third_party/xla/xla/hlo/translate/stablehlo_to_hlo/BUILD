load("//xla/tsl:tsl.bzl", "internal_visibility")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = internal_visibility([
        "//learning/brain/mlir:machina_friends",
        "//learning/brain/mlir:xla_friends",
    ]),
    licenses = ["notice"],
)

cc_library(
    name = "translate",
    srcs = ["translate.cc"],
    hdrs = ["translate.h"],
    deps = [
        "//xla/hlo/translate:register",
        "//xla/hlo/translate/mhlo_to_hlo:translate",
        "//xla/mlir_hlo:mhlo_passes",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Parser",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:Support",
    ],
)

cc_library(
    name = "translate_registration",
    testonly = True,
    srcs = ["translate_registration.cc"],
    deps = [
        ":translate",
        "//xla/hlo/translate:register",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:ArithDialect",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Support",
        "@toolchain-project//mlir:TensorDialect",
        "@toolchain-project//mlir:TranslateLib",
        "@stablehlo//:register",
    ],
    alwayslink = 1,
)

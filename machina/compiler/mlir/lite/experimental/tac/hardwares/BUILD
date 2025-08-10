load("//machina/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "gpu_hardware",
    srcs = ["gpu_hardware.cc"],
    hdrs = ["gpu_hardware.h"],
    deps = [
        ":target_hardware",
        "//machina/compiler/mlir/lite:cost_estimators",
        "//machina/compiler/mlir/lite:machina_lite",
        "//machina/compiler/mlir/lite/experimental/tac:common",
        "//machina/compiler/mlir/lite/experimental/tac:device_transform_patterns",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Support",
    ],
    alwayslink = 1,
)

cc_library(
    name = "cpu_hardware",
    srcs = ["cpu_hardware.cc"],
    deps = [
        ":target_hardware",
        "//machina/compiler/mlir/lite:cost_estimators",
        "//machina/compiler/mlir/lite:machina_lite",
        "//machina/compiler/mlir/lite/experimental/tac:common",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Support",
    ],
    alwayslink = 1,
)

cc_library(
    name = "nnapi_hardware",
    srcs = ["nnapi_hardware.cc"],
    hdrs = ["nnapi_hardware.h"],
    deps = [
        ":target_hardware",
        "//machina/compiler/mlir/lite:cost_estimators",
        "//machina/compiler/mlir/lite:machina_lite",
        "//machina/compiler/mlir/lite/experimental/tac:device_transform_patterns",
        "//machina/compiler/mlir/lite/experimental/tac/hardwares:simple_hardware",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Support",
    ],
    alwayslink = 1,
)

cc_library(
    name = "target_hardware",
    srcs = ["target_hardware.cc"],
    hdrs = ["target_hardware.h"],
    deps = [
        "//machina/compiler/mlir/lite:machina_lite",
        "//machina/compiler/mlir/lite/experimental/tac:common",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Support",
    ],
)

cc_library(
    name = "simple_hardware",
    srcs = ["simple_hardware.cc"],
    hdrs = ["simple_hardware.h"],
    deps = [
        ":target_hardware",
        "@toolchain-project//mlir:IR",
    ],
)

cc_library(
    name = "all-target-hardwares",
    deps = [
        ":cpu_hardware",
        ":gpu_hardware",
        ":nnapi_hardware",
    ],
    alwayslink = 1,
)

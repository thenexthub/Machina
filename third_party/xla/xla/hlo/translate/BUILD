load("@bazel_skylib//rules:build_test.bzl", "build_test")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//xla:xla.default.bzl", "xla_cc_binary")
load("//xla/tsl:tsl.bzl", "internal_visibility")
load("//xla/tsl:tsl.default.bzl", "get_compatible_with_portable")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = internal_visibility([
        "//learning/brain/mlir:machina_friends",
        "//learning/brain/mlir:xla_friends",
    ]),
    licenses = ["notice"],
)

cc_library(
    name = "portable_api",
    srcs = ["portable_api.cc"],
    hdrs = ["portable_api.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":register",
        ":stablehlo",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/translate/hlo_to_mhlo:hlo_to_mlir_hlo",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:BytecodeWriter",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Support",
    ],
)

build_test(
    name = "xla-translate_build_test",
    targets = [
        ":xla-translate",
    ],
)

xla_cc_binary(
    name = "xla-translate",
    testonly = True,
    srcs = ["xla_translate_main.cc"],
    deps = [
        "//xla/hlo/translate/hlo_to_mhlo:translate_registration",
        "//xla/hlo/translate/mhlo_to_hlo:translate_registration",
        "//xla/hlo/translate/stablehlo_to_hlo:translate_registration",
        "//xla/service/cpu:cpu_compiler",
        "//xla/service/cpu:cpu_transfer_manager",
        "//xla/stream_executor/host:host_platform",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Support",
        "@toolchain-project//mlir:TranslateLib",
        "@local_tsl//tsl/platform:platform_port",
    ],
)

build_test(
    name = "xla-translate-opt_build_test",
    targets = [
        ":xla-translate-opt",
    ],
)

xla_cc_binary(
    name = "xla-translate-opt",
    testonly = True,
    srcs = ["xla_translate_opt_main.cc"],
    deps = [
        ":register",
        "//xla/mlir/framework/ir:xla_framework",
        "//xla/mlir/framework/transforms:passes",
        "//xla/service:cpu_plugin",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:AllPassesAndDialects",
        "@toolchain-project//mlir:MlirOptLib",
        "@toolchain-project//mlir:RegisterAllDialects",
        "@toolchain-project//mlir:RegisterAllPasses",
        "@local_tsl//tsl/platform:platform_port",
    ],
)

cc_library(
    name = "register",
    srcs = ["register.cc"],
    hdrs = ["register.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//xla/mlir_hlo:hlo_dialect_registration",
        "@toolchain-project//mlir:ArithDialect",
        "@toolchain-project//mlir:FuncExtensions",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:ShapeDialect",
        "@toolchain-project//mlir:TensorDialect",
        "@toolchain-project//mlir:UBDialect",
        "@shardy//shardy/dialect/sdy/ir:register",
        "@stablehlo//:register",
    ],
)

cc_library(
    name = "stablehlo",
    srcs = ["stablehlo.cc"],
    hdrs = ["stablehlo.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//xla:debug_options_flags",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/translate/hlo_to_mhlo:hlo_module_importer",
        "//xla/hlo/translate/mhlo_to_hlo:mlir_hlo_to_hlo",
        "//xla/hlo/translate/mhlo_to_hlo:module_attributes_exporter",
        "//xla/mlir/utils:error_util",
        "//xla/mlir_hlo:mhlo_passes",
        "//xla/mlir_hlo:stablehlo_extension_passes",
        "//xla/service:hlo_proto_cc",
        "//xla/service/llvm_ir:llvm_util",
        "//xla/tsl/platform:errors",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:ArithDialect",
        "@toolchain-project//mlir:FuncExtensions",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:ShapeDialect",
        "@toolchain-project//mlir:TensorDialect",
        "@toolchain-project//mlir:Transforms",
        "@toolchain-project//mlir:UBDialect",
        "@stablehlo//:stablehlo_passes",
    ],
)

cc_library(
    name = "attributes",
    hdrs = ["attributes.h"],
    compatible_with = get_compatible_with_portable(),
)

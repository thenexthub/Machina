load("@bazel_skylib//rules:common_settings.bzl", "bool_flag")
load("//machina:machina.bzl", "tf_cc_test")
load("//machina:machina.default.bzl", "filegroup")
load("//machina/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

bool_flag(
    name = "disable_mlir",
    build_setting_default = False,
)

config_setting(
    name = "disable_mlir_config",
    flag_values = {":disable_mlir": "True"},
    visibility = ["//visibility:public"],
)

cc_library(
    name = "mlir",
    srcs = ["mlir.cc"],
    hdrs = ["mlir.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "//machina/cc/saved_model:bundle_v2",
        "//machina/cc/saved_model:loader",
        "//machina/compiler/mlir/machina/translate/tools:parsers",
        "//machina/compiler/mlir/quantization/stablehlo:bridge_passes",
        "@com_google_absl//absl/strings",
        "@toolchain-project//mlir:FuncExtensions",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:AllPassesAndDialects",
        "@toolchain-project//mlir:BytecodeWriter",
        "@toolchain-project//mlir:ShapeDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Parser",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:RegisterAllPasses",
        "//machina/core/common_runtime:graph_constructor",
        "@toolchain-project//mlir:Support",
        "@stablehlo//:register",
        "//machina/c:tf_status",
        "//machina/c:tf_status_helper",
        "//machina/compiler/mlir/tf2xla/api/v2:graph_to_tf_executor",
        "//machina/c/eager:c_api",
        "//machina/c/eager:tfe_context_internal",
        "//machina/compiler/mlir/machina:mlir_import_options",
        "@local_xla//xla/mlir/framework/transforms:passes",
        "@local_xla//xla/mlir_hlo:all_passes",
        "//machina/compiler/mlir/machina",
        "//machina/compiler/mlir/machina:import_model",
        "//machina/compiler/mlir/machina:error_util",
        "//machina/compiler/mlir/machina:import_utils",
        "//machina/compiler/mlir/machina:mlir_roundtrip_flags",
        "//machina/compiler/mlir/machina:mlprogram_util",
        "//machina/compiler/mlir/machina/transforms:machina_passes",
        "//machina/compiler/mlir/machina/transforms:tf_saved_model_passes",
        "//machina/compiler/mlir/machina:translate_lib",
        "//machina/compiler/mlir/tf2xla/transforms:xla_legalize_tf",
        "//machina/core:framework",
        "//machina/core:lib_proto_parsing",
        "//machina/core:protos_all_cc",
        "//machina/core:tflite_portable_logging",
        "//machina/core/common_runtime/eager:context",
        # (yongtang) The graph_optimization_pass_registration needs to be part
        # of a shared object that will be loaded whenever `import machina`
        # is run. The natural place is libmachina_framework.so.
        # While adding graph_optimization_pass_registration to
        # libmachina_framework.so is possible with some modification in
        # dependency, many tests will fail due to multiple copies of LLVM.
        # See https://github.com/machina/machina/pull/39231 for details.
        # Alternatively, we place graph_optimization_pass_registration here
        # because:
        # - machina/python/_pywrap_mlir.so already depends on LLVM anyway
        # - machina/python/_pywrap_mlir.so always loaded as part of python
        #   binding
        # TODO: It might be still preferrable to place graph_optimization_pass
        # as part of the libmachina_framework.so, as it is the central
        # place for core related components.
        "//machina/compiler/mlir/machina/transforms:graph_optimization_pass_registration",  # buildcleaner: keep
    ],
    alwayslink = 1,
)

tf_cc_test(
    name = "mlir_test",
    srcs = ["mlir_test.cc"],
    deps = [
        ":mlir",
        "//machina/c:safe_ptr",
        "//machina/c:tf_status_headers",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
    ],
)

filegroup(
    name = "pywrap_mlir_hdrs",
    srcs = [
        "mlir.h",
    ],
    visibility = [
        "//machina/python:__pkg__",
    ],
)

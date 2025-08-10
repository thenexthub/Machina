load("@flatbuffers//:build_defs.bzl", "flatbuffer_cc_library")
load(
    "@toolchain-project//mlir:tblgen.bzl",
    "gentbl_cc_library",
)
load("//machina:strict.default.bzl", "py_strict_library")
load("//machina:machina.bzl", "if_google", "tf_cc_binary", "tf_cc_test")
load("//machina:machina.default.bzl", "get_compatible_with_portable")
load(
    "//machina/core/platform:build_config.bzl",
    "tf_proto_library",
)
load("//machina/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

flatbuffer_cc_library(
    name = "runtime_metadata_fbs",
    srcs = ["runtime_metadata.fbs"],
    compatible_with = get_compatible_with_portable(),
)

cc_library(
    name = "execution_metadata_exporter",
    srcs = [
        "execution_metadata_exporter.cc",
    ],
    hdrs = [
        "execution_metadata_exporter.h",
    ],
    deps = [
        ":common",
        ":runtime_metadata_fbs",
        "//machina/compiler/mlir/lite:machina_lite",
        "//machina/compiler/mlir/lite/experimental/tac/hardwares:target_hardware",
        "//machina/compiler/mlir/lite/quantization/ir:QuantOps",
        "//machina/compiler/mlir/machina",
        "@flatbuffers",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:ArithDialect",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Support",
    ],
)

tf_cc_test(
    name = "execution_metadata_exporter_test",
    srcs = [
        "execution_metadata_exporter_test.cc",
    ],
    deps = [
        ":execution_metadata_exporter",
        ":runtime_metadata_fbs",
        "//machina/compiler/mlir/lite:machina_lite",
        "//machina/compiler/mlir/lite/experimental/tac/hardwares:all-target-hardwares",
        "@com_google_googletest//:gtest_main",
        "@flatbuffers",
        "@toolchain-project//mlir:ArithDialect",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Parser",
    ],
)

cc_library(
    name = "common",
    srcs = [
        "common/utils.cc",
    ],
    hdrs = [
        "common/cost.h",
        "common/subgraph.h",
        "common/targets.h",
        "common/utils.h",
    ],
    deps = [
        "//machina/compiler/mlir/lite:machina_lite",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:ArithDialect",
        "@toolchain-project//mlir:BytecodeOpInterface",
        "@toolchain-project//mlir:CallOpInterfaces",
        "@toolchain-project//mlir:CastInterfaces",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:QuantOps",
        "@toolchain-project//mlir:SideEffectInterfaces",
        "@toolchain-project//mlir:Support",
    ],
)

gentbl_cc_library(
    name = "transform_patterns_inc_gen",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = {"transforms/generated_transform_patterns.inc": ["-gen-rewriters"]},
    tblgen = "@toolchain-project//mlir:mlir-tblgen",
    td_file = "transforms/transform_patterns.td",
    deps = [
        "//machina/compiler/mlir/lite:machina_lite_ops_td_files",
        "//machina/compiler/mlir/machina:machina_ops_td_files",
        "@toolchain-project//mlir:ArithOpsTdFiles",
        "@toolchain-project//mlir:FuncTdFiles",
    ],
)

cc_library(
    name = "device_transform_patterns",
    srcs = [
        "transforms/device_transform_patterns.cc",
    ],
    hdrs = [
        "transforms/device_transform_patterns.h",
    ],
    textual_hdrs = [
        "transforms/generated_transform_patterns.inc",
    ],
    deps = [
        ":common",
        "//machina/compiler/mlir/lite:machina_lite",
        "//machina/compiler/mlir/machina",
        "//machina/compiler/mlir/machina:verification_utils",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:QuantOps",
        "@toolchain-project//mlir:Support",
    ],
)

cc_library(
    name = "device_transform_gpu",
    srcs = [
        "transforms/device_transform_gpu.cc",
    ],
    hdrs = [
        "transforms/device_transform_gpu.h",
    ],
    deps = [
        ":common",
        "//machina/compiler/mlir/lite:machina_lite",
        "//machina/compiler/mlir/lite/experimental/tac/hardwares:gpu_hardware",
        "//machina/compiler/mlir/machina",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:Support",
        "@toolchain-project//mlir:TransformUtils",
    ],
    alwayslink = 1,
)

cc_library(
    name = "device_transform_nnapi",
    srcs = [
        "transforms/device_transform_nnapi.cc",
    ],
    deps = [
        ":common",
        "//machina/compiler/mlir/lite:machina_lite",
        "//machina/compiler/mlir/lite/experimental/tac/hardwares:nnapi_hardware",
        "//machina/compiler/mlir/machina",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:Support",
        "@toolchain-project//mlir:TransformUtils",
    ],
    alwayslink = 1,
)

cc_library(
    name = "device_transform",
    srcs = [
        "transforms/device_transform.cc",
    ],
    hdrs = [
        "transforms/device_transform.h",
    ],
    deps = [
        ":common",
        ":device_transform_gpu",
        ":device_transform_nnapi",
        ":device_transform_patterns",
        "//machina/compiler/mlir/lite:machina_lite",
        "//machina/compiler/mlir/lite/experimental/tac/hardwares:target_hardware",
        "//machina/compiler/mlir/machina",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:QuantOps",
        "@toolchain-project//mlir:Support",
        "@toolchain-project//mlir:TransformUtils",
        "@toolchain-project//mlir:Transforms",
    ],
)

cc_library(
    name = "cost_model",
    srcs = [
        "transforms/cost_model.cc",
    ],
    hdrs = [
        "transforms/cost_model.h",
    ],
    deps = [
        ":common",
        "//machina/compiler/mlir/lite:machina_lite",
        "//machina/compiler/mlir/lite/experimental/tac/hardwares:target_hardware",
        "@com_google_absl//absl/strings",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:CallOpInterfaces",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:Support",
    ],
    alwayslink = 1,
)

# TODO(b/177376459): split tac_module and passes dependency to separate libraries.
cc_library(
    name = "target_aware_conversion",
    srcs = [
        "tac_module.cc",
        "transforms/compute_cost.cc",
        "transforms/fold_constants_to_subgraph.cc",
        "transforms/get_alternative_subgraph.cc",
        "transforms/pick_subgraphs.cc",
        "transforms/raise_target_subgraphs.cc",
        "transforms/tac_filter.cc",
        "transforms/target_annotation.cc",
    ],
    hdrs = [
        "tac_module.h",
        "transforms/passes.h",
        "transforms/tac_pass.h",
    ],
    deps = [
        ":common",
        ":cost_model",
        ":device_transform",
        ":tac_filter_proto_cc",
        ":tac_importer_exporter",
        "//machina/compiler/mlir/lite:machina_lite",
        "//machina/compiler/mlir/lite:tf_tfl_passes",
        "//machina/compiler/mlir/lite/experimental/common:outline_operations",
        "//machina/compiler/mlir/lite/experimental/tac/hardwares:target_hardware",
        "//machina/compiler/mlir/machina:cluster_util",
        "//machina/compiler/mlir/machina:error_util",
        "//machina/compiler/mlir/machina:machina_analysis",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_protobuf//:protobuf_headers",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:ArithDialect",
        "@toolchain-project//mlir:CallOpInterfaces",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:SideEffectInterfaces",
        "@toolchain-project//mlir:Support",
        "@toolchain-project//mlir:TransformUtils",
        "@toolchain-project//mlir:Transforms",
    ],
    alwayslink = 1,
)

cc_library(
    name = "tac-opt_lib",
    testonly = True,
    deps = [
        ":target_aware_conversion",
        "//machina/compiler/mlir/lite:litert_mlir_opt_main",
    ],
    alwayslink = 1,
)

# Binary with no hardwares linked.
tf_cc_binary(
    name = "tac-opt",
    testonly = True,
    deps = [
        ":tac-opt_lib",
    ],
)

# Binary with all backends linked.
tf_cc_binary(
    name = "tac-opt-all-backends",
    testonly = True,
    deps = [
        ":tac-opt_lib",
        "//machina/compiler/mlir/lite/experimental/common:outline_operations",
        "//machina/compiler/mlir/lite/experimental/tac/hardwares:all-target-hardwares",
    ],
)

cc_library(
    name = "tac-translate-lib",
    srcs = [
        "tac_translate.cc",
    ],
    deps = [
        ":common",
        ":execution_metadata_exporter",
        ":target_aware_conversion",
        ":tflite_importer_exporter",
        "//machina/compiler/mlir:init_mlir",
        "//machina/compiler/mlir/lite:machina_lite_legalize_tf",
        "//machina/compiler/mlir/lite:machina_lite_optimize",
        "//machina/compiler/mlir/lite/experimental/tac/hardwares:target_hardware",
        "//machina/compiler/mlir/lite/experimental/tac/utils",
        "//machina/compiler/mlir/machina",
        "//machina/compiler/mlir/machina:error_util",
        "//machina/compiler/mlir/machina/transforms:machina_passes",
        "//machina/core:lib",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Parser",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:Support",
        "@toolchain-project//mlir:Transforms",
        "@toolchain-project//mlir:TranslateLib",
    ],
)

tf_cc_binary(
    name = "tac-translate",
    deps = [
        ":tac-translate-lib",
        "//machina/compiler/mlir/lite/experimental/tac/hardwares:all-target-hardwares",
    ],
)

cc_library(
    name = "tac_importer_exporter",
    hdrs = ["tac_importer_exporter.h"],
    deps = [
        "@com_google_absl//absl/status:statusor",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
    ],
)

cc_library(
    name = "tflite_importer_exporter",
    srcs = ["tflite_import_export.cc"],
    hdrs = ["tflite_import_export.h"],
    deps = [
        ":common",
        ":execution_metadata_exporter",
        ":tac_importer_exporter",
        "//machina/compiler/mlir/lite/experimental/tac/hardwares:target_hardware",
        "//machina/compiler/mlir/lite/experimental/tac/utils",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:Support",
    ],
)

exports_files([
    "run_lit.sh",
])

py_strict_library(
    name = "tac",
    srcs = [
        "tac.py",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//machina/compiler/mlir/lite/experimental/tac/py_wrapper:_pywrap_tac_wrapper",
    ],
)

tf_proto_library(
    name = "tac_filter_proto",
    srcs = ["tac_filter.proto"],
    protodeps = if_google(["//google/protobuf:any"]),
    visibility = ["//visibility:public"],
)

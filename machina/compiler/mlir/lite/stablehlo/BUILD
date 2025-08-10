load("@toolchain-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")
load("//machina:machina.bzl", "tf_cc_binary")
load("//machina:machina.default.bzl", "get_compatible_with_portable")
load("//machina/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "drop_savedmodel_semantics",
    srcs = [
        "transforms/drop_savedmodel_semantics.cc",
    ],
    hdrs = [
        "transforms/drop_savedmodel_semantics.h",
    ],
    copts = [
        "-Ithird_party",
    ],
    deps = [
        ":stablehlo_util",
        "//machina/compiler/mlir/machina",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Pass",
    ],
    alwayslink = 1,
)

cc_library(
    name = "stablehlo_util",
    srcs = [
        "transforms/stablehlo_util.cc",
    ],
    hdrs = [
        "transforms/stablehlo_util.h",
    ],
    copts = [
        "-Ithird_party",
    ],
    deps = [
        "@toolchain-project//toolchain:Support",
    ],
    alwayslink = 1,
)

cc_library(
    name = "hlo_matchers",
    srcs = [
        "transforms/hlo_matchers.cc",
    ],
    hdrs = [
        "transforms/hlo_matchers.h",
    ],
    copts = [
        "-Ithird_party",
    ],
    deps = [
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Support",
        "@toolchain-project//mlir:TransformUtils",
        "@toolchain-project//mlir:Transforms",
        "@local_xla//xla/mlir_hlo",
    ],
)

cc_library(
    name = "smuggle_disallowed_ops",
    srcs = [
        "transforms/smuggle_disallowed_ops.cc",
    ],
    hdrs = [
        "transforms/smuggle_disallowed_ops.h",
    ],
    copts = [
        "-Ithird_party",
    ],
    deps = [
        ":stablehlo_util",
        "//machina/compiler/mlir/machina",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:TransformUtils",
        "@toolchain-project//mlir:Transforms",
        "@stablehlo//:stablehlo_ops",
    ],
    alwayslink = 1,
)

cc_library(
    name = "tfl_stablehlo",
    srcs = [
        "transforms/tfl_stablehlo_pass.cc",
    ],
    hdrs = [
        "transforms/tfl_stablehlo_pass.h",
    ],
    copts = [
        "-Ithird_party",
    ],
    deps = [
        "//machina/compiler/mlir/lite:machina_lite",
        "//machina/compiler/mlir/machina:dynamic_shape_utils",
        "@flatbuffers",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:Support",
        "@toolchain-project//mlir:TransformUtils",
        "@stablehlo//:register",
        "@stablehlo//:stablehlo_ops",
    ],
    alwayslink = 1,
)

cc_library(
    name = "transforms",
    srcs = [
        "transforms/transforms.cc",
    ],
    hdrs = [
        "transforms/transforms.h",
    ],
    copts = [
        "-Ithird_party",
    ],
    deps = [
        ":drop_savedmodel_semantics",
        ":legalize_stablehlo_custom_call_to_composite",
        ":legalize_tf_xla_call_module_to_stablehlo_pass",
        ":optimize",
        ":smuggle_disallowed_ops",
        ":stablehlo_fuse_convolution_pass",
        ":stablehlo_unfuse_batch_norm_pass",
        ":unfold_splat_constant_pass",
        "//machina/compiler/mlir/quantization/stablehlo:bridge_passes",
        "//machina/compiler/mlir/stablehlo:fold_broadcast_pass",
        "//machina/compiler/mlir/stablehlo:fuse_convolution_pass",
        "//machina/compiler/mlir/stablehlo:rename_entrypoint_to_main",
        "//machina/compiler/mlir/stablehlo:tf_stablehlo",
        "//machina/compiler/mlir/stablehlo:unfuse_batch_norm_pass",
        "//machina/compiler/mlir/machina/transforms:machina_passes",
        "//machina/compiler/mlir/machina/transforms:tf_saved_model_passes",
        "//machina/compiler/mlir/tf2xla/transforms:tf_xla_passes",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:Transforms",
        "@local_xla//xla/mlir_hlo:mhlo_passes",
    ],
)

cc_library(
    name = "check_accepted_ops_pass",
    srcs = [
        "transforms/check_accepted_ops_pass.cc",
    ],
    hdrs = [
        "transforms/check_accepted_ops_pass.h",
    ],
    copts = [
        "-Ithird_party",
    ],
    deps = [
        ":stablehlo_util",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:Transforms",
    ],
)

cc_library(
    name = "op_stat_pass",
    srcs = [
        "transforms/op_stat_pass.cc",
    ],
    hdrs = [
        "transforms/op_stat_pass.h",
    ],
    copts = [
        "-Ithird_party",
    ],
    deps = [
        ":stablehlo_util",
        "//machina/compiler/mlir/lite:machina_lite",
        "//machina/compiler/mlir/lite/quantization/common/quantization_lib",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:QuantOps",
        "@toolchain-project//mlir:Support",
        "@toolchain-project//mlir:Transforms",
    ],
)

gentbl_cc_library(
    name = "passes_inc_gen",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = {"transforms/stablehlo_passes.h.inc": [
        "-gen-pass-decls",
        "-name=OdmlStablehlo",
    ]},
    tblgen = "@toolchain-project//mlir:mlir-tblgen",
    td_file = "transforms/stablehlo_passes.td",
    deps = ["@toolchain-project//mlir:PassBaseTdFiles"],
)

cc_library(
    name = "compose_uniform_quantized_type_pass",
    srcs = ["transforms/compose_uniform_quantized_type_pass.cc"],
    hdrs = ["transforms/stablehlo_passes.h"],
    copts = ["-Ithird_party"],
    deps = [
        ":passes_inc_gen",
        "//machina/compiler/mlir/quantization/common:attrs_and_constraints",
        "//machina/compiler/mlir/quantization/common:uniform_quantized_types",
        "@com_google_absl//absl/algorithm:container",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:QuantOps",
        "@toolchain-project//mlir:Support",
        "@toolchain-project//mlir:TransformUtils",
        "@stablehlo//:stablehlo_ops",
    ],
    alwayslink = 1,
)

cc_library(
    name = "stablehlo_unfuse_batch_norm_pass",
    srcs = [
        "transforms/stablehlo_unfuse_batch_norm_pass.cc",
    ],
    hdrs = [
        "transforms/stablehlo_passes.h",
    ],
    copts = [
        "-Ithird_party",
    ],
    deps = [
        ":passes_inc_gen",
        "//machina/compiler/mlir/machina",
        "//machina/compiler/mlir/machina:machina_ops",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:ShapeDialect",
        "@toolchain-project//mlir:Support",
        "@toolchain-project//mlir:TransformUtils",
        "@stablehlo//:stablehlo_ops",
    ],
    alwayslink = 1,
)

cc_library(
    name = "stablehlo_fuse_convolution_pass",
    srcs = [
        "transforms/stablehlo_fuse_convolution_pass.cc",
    ],
    hdrs = [
        "transforms/stablehlo_passes.h",
    ],
    copts = [
        "-Ithird_party",
    ],
    deps = [
        ":passes_inc_gen",
        "//machina/compiler/mlir/lite:validators",
        "//machina/compiler/mlir/quantization/common:attrs_and_constraints",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:Dialect",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:ShapeDialect",
        "@toolchain-project//mlir:Support",
        "@toolchain-project//mlir:TransformUtils",
        "@stablehlo//:stablehlo_ops",
    ],
    alwayslink = 1,
)

cc_library(
    name = "unfold_splat_constant_pass",
    srcs = [
        "transforms/unfold_splat_constant_pass.cc",
    ],
    hdrs = [
        "transforms/stablehlo_passes.h",
    ],
    copts = [
        "-Ithird_party",
    ],
    deps = [
        ":passes_inc_gen",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:QuantOps",
        "@toolchain-project//mlir:Support",
        "@toolchain-project//mlir:Transforms",
        "@local_xla//xla/mlir_hlo",
    ],
    alwayslink = 1,
)

cc_library(
    name = "legalize_tf_xla_call_module_to_stablehlo_pass",
    srcs = [
        "transforms/legalize_tf_xla_call_module_to_stablehlo_pass.cc",
    ],
    hdrs = [
        "transforms/legalize_tf_xla_call_module_to_stablehlo_pass.h",
    ],
    copts = [
        "-Ithird_party",
    ],
    deps = [
        "//machina/compiler/mlir/machina",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:QuantOps",
        "@toolchain-project//mlir:ShapeDialect",
        "@toolchain-project//mlir:Support",
        "@toolchain-project//mlir:TransformUtils",
        "@toolchain-project//mlir:Transforms",
        "@stablehlo//:stablehlo_ops",
        "@stablehlo//:stablehlo_serialization",
        "@stablehlo//:vhlo_ops",
    ],
    alwayslink = 1,
)

cc_library(
    name = "legalize_stablehlo_composite_to_tfl_custom",
    srcs = [
        "transforms/legalize_stablehlo_composite_to_tfl_custom.cc",
    ],
    hdrs = [
        "transforms/stablehlo_passes.h",
        "transforms/stablehlo_passes.h.inc",
    ],
    copts = [
        "-Ithird_party",
    ],
    deps = [
        "//machina/compiler/mlir/lite:machina_lite",
        "@flatbuffers",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:Support",
        "@toolchain-project//mlir:TransformUtils",
        "@stablehlo//:stablehlo_ops",
    ],
    alwayslink = 1,
)

cc_library(
    name = "legalize_stablehlo_to_vhlo_pass",
    srcs = [
        "transforms/legalize_stablehlo_to_vhlo.cc",
    ],
    hdrs = [
        "transforms/stablehlo_passes.h",
        "transforms/stablehlo_passes.h.inc",
    ],
    copts = [
        "-Ithird_party",
    ],
    deps = [
        ":uniform_quantized_stablehlo_to_tfl_pass",
        "//machina/compiler/mlir/lite:machina_lite",
        "//machina/compiler/mlir/lite/core:macros",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:ArithDialect",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:InferTypeOpInterface",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:QuantOps",
        "@toolchain-project//mlir:ReconcileUnrealizedCasts",
        "@toolchain-project//mlir:ShapeDialect",
        "@toolchain-project//mlir:Support",
        "@toolchain-project//mlir:TensorDialect",
        "@toolchain-project//mlir:TransformUtils",
        "@toolchain-project//mlir:Transforms",
        "@stablehlo//:base",
        "@stablehlo//:stablehlo_ops",
        "@stablehlo//:stablehlo_ops_inc_gen",
        "@stablehlo//:stablehlo_pass_inc_gen",
        "@stablehlo//:stablehlo_passes",
        "@stablehlo//:stablehlo_portable_api",
        "@stablehlo//:stablehlo_type_inference",
        "@stablehlo//:version",
        "@stablehlo//:vhlo_ops",
        "@stablehlo//:vhlo_types",
    ],
    alwayslink = 1,
)

cc_library(
    name = "legalize_stablehlo_custom_call_to_composite",
    srcs = [
        "transforms/legalize_stablehlo_custom_call_to_composite.cc",
    ],
    hdrs = [
        "transforms/stablehlo_passes.h",
        "transforms/stablehlo_passes.h.inc",
    ],
    copts = [
        "-Ithird_party",
    ],
    deps = [
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:Support",
        "@toolchain-project//mlir:TransformUtils",
        "@toolchain-project//mlir:Transforms",
        "@stablehlo//:stablehlo_ops",
    ],
    alwayslink = 1,
)

cc_library(
    name = "optimize",
    srcs = [
        "transforms/optimize.cc",
    ],
    hdrs = [
        "transforms/stablehlo_passes.h",
    ],
    copts = [
        "-Ithird_party",
    ],
    deps = [
        ":passes_inc_gen",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:Support",
        "@toolchain-project//mlir:TransformUtils",
        "@local_xla//xla/mlir_hlo",
    ],
    alwayslink = 1,
)

cc_library(
    name = "uniform_quantized_stablehlo_to_tfl_pass",
    srcs = ["transforms/uniform_quantized_stablehlo_to_tfl_pass.cc"],
    hdrs = [
        "transforms/stablehlo_passes.h",
        "transforms/stablehlo_passes.h.inc",
    ],
    copts = ["-Ithird_party"],
    deps = [
        ":passes_inc_gen",
        "//machina/compiler/mlir/lite:machina_lite",
        "//machina/compiler/mlir/quantization/common:attrs_and_constraints",
        "//machina/compiler/mlir/quantization/common:uniform_quantized_types",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/log:check",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:ArithDialect",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:QuantOps",
        "@toolchain-project//mlir:Support",
        "@toolchain-project//mlir:TransformUtils",
        "@stablehlo//:base",
        "@stablehlo//:stablehlo_ops",
    ],
    alwayslink = 1,
)

gentbl_cc_library(
    name = "hlo_legalize_tf_inc_gen",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = {"transforms/generated_legalize_hlo.inc": ["-gen-rewriters"]},
    tblgen = "@toolchain-project//mlir:mlir-tblgen",
    td_file = "transforms/legalize_hlo_patterns.td",
    deps = [
        "//machina/compiler/mlir/machina:machina_ops_td_files",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:FuncTdFiles",
        "@local_xla//xla/mlir_hlo:hlo_ops_td_files",
    ],
)

gentbl_cc_library(
    name = "hlo_legalize_tflite_inc_gen",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = {"transforms/generated_tflite_legalize_hlo.inc": ["-gen-rewriters"]},
    tblgen = "@toolchain-project//mlir:mlir-tblgen",
    td_file = "transforms/tflite_legalize_hlo_patterns.td",
    deps = [
        "//machina/compiler/mlir/lite:machina_lite_ops_td_files",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:FuncTdFiles",
        "@local_xla//xla/mlir_hlo:hlo_ops_td_files",
    ],
)

cc_library(
    name = "tfl_legalize_hlo",
    srcs = [
        "transforms/tflite_legalize_hlo.cc",
    ],
    hdrs = [
        "transforms/stablehlo_passes.h",
    ],
    copts = [
        "-Ithird_party",
    ],
    deps = [
        ":hlo_legalize_tflite_inc_gen",
        ":passes_inc_gen",
        ":unfold_splat_constant_pass",
        "//machina/compiler/mlir/lite:machina_lite",
        "//machina/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions:conv",
        "//machina/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions:custom_call",
        "//machina/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions:dot_general",
        "//machina/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions:fft",
        "//machina/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions:gather",
        "//machina/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions:get_dimension_size",
        "//machina/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions:if",
        "//machina/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions:iota",
        "//machina/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions:pad",
        "//machina/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions:reduce",
        "//machina/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions:reduce_window",
        "//machina/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions:slice",
        "//machina/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions:sort",
        "//machina/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions:util",
        "//machina/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions:while",
        "//machina/compiler/mlir/machina",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:ArithDialect",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:Support",
        "@toolchain-project//mlir:TransformUtils",
        "@local_xla//xla/mlir_hlo",
    ],
    alwayslink = True,
)

gentbl_cc_library(
    name = "prepare_hlo_inc_gen",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = {"transforms/generated_prepare_hlo.inc": ["-gen-rewriters"]},
    tblgen = "@toolchain-project//mlir:mlir-tblgen",
    td_file = "transforms/prepare_hlo.td",
    deps = [
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:FuncTdFiles",
        "@local_xla//xla/mlir_hlo:hlo_ops_td_files",
    ],
)

cc_library(
    name = "prepare_hlo",
    srcs = [
        "transforms/prepare_hlo.cc",
    ],
    hdrs = [
        "transforms/stablehlo_passes.h",
    ],
    copts = [
        "-Ithird_party",
    ],
    deps = [
        ":passes_inc_gen",
        ":prepare_hlo_inc_gen",
        "//machina/compiler/mlir/lite:machina_lite_ops",
        "//machina/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions:conv",
        "//machina/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions:conv_util",
        "//machina/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions:custom_call",
        "//machina/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions:fft",
        "//machina/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions:pad_util",
        "//machina/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions:reduce_window",
        "//machina/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions:slice",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:Support",
        "@toolchain-project//mlir:TransformUtils",
        "@local_xla//xla/mlir_hlo",
    ],
    alwayslink = True,
)

cc_library(
    name = "tfl_legalize_chlo",
    srcs = [
        "transforms/tfl_legalize_chlo.cc",
    ],
    hdrs = ["transforms/stablehlo_passes.h"],
    compatible_with = get_compatible_with_portable(),
    copts = [
        "-Ithird_party",
    ],
    deps = [
        ":passes_inc_gen",
        "//machina/compiler/mlir/lite:machina_lite",
        "//machina/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions:gelu",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:Support",
        "@toolchain-project//mlir:TransformUtils",
        "@stablehlo//:chlo_ops",
        "@stablehlo//:stablehlo_ops",
    ],
    alwayslink = 1,
)

cc_library(
    name = "tf_legalize_hlo",
    srcs = [
        "transforms/generated_legalize_hlo.inc",
        "transforms/legalize_hlo.cc",
    ],
    hdrs = [
        "transforms/stablehlo_passes.h",
        "transforms/stablehlo_passes.h.inc",
    ],
    copts = [
        "-Ithird_party",
    ],
    deps = [
        ":hlo_legalize_tf_inc_gen",
        ":hlo_matchers",
        ":passes_inc_gen",
        "//machina/compiler/mlir/lite:machina_lite",
        "//machina/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions:reduce",
        "//machina/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions:util",
        "//machina/compiler/mlir/machina",
        "//machina/compiler/mlir/machina:dynamic_shape_utils",
        "//machina/compiler/mlir/machina/transforms:lower_tf_lib",
        "//machina/compiler/mlir/machina/transforms:tf_device_pass_inc_gen",
        "//machina/core:framework",
        "//machina/core:lib",
        "@com_google_absl//absl/status",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:Analysis",
        "@toolchain-project//mlir:ArithDialect",
        "@toolchain-project//mlir:Dialect",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:QuantOps",
        "@toolchain-project//mlir:Support",
        "@toolchain-project//mlir:TransformUtils",
        "@toolchain-project//mlir:Transforms",
        "@local_xla//xla/mlir_hlo",
        "@stablehlo//:broadcast_utils",
        "@stablehlo//:chlo_ops",
    ],
)

td_library(
    name = "composite_td_files",
    srcs = [
        "transforms/composite_avg_pool_patterns.td",
        "transforms/composite_utils.td",
    ],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//machina/compiler/mlir/lite:machina_lite_ops_td_files",
        "@toolchain-project//mlir:FuncTdFiles",
        "@local_xla//xla/mlir_hlo:hlo_ops_td_files",
    ],
)

cc_library(
    name = "composite_utils",
    srcs = ["transforms/composite_utils.cc"],
    hdrs = ["transforms/composite_utils.h"],
    deps = [
        "//machina/compiler/mlir/lite:machina_lite",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Support",
        "@toolchain-project//mlir:TransformUtils",
        "@local_xla//xla/mlir_hlo",
    ],
)

cc_library(
    name = "lift_callsite_loc_caller",
    srcs = ["transforms/torch/lift_callsite_loc_caller_pass.cc"],
    copts = ["-Ithird_party"],
    deps = [
        ":passes_inc_gen",
        ":prepare_hlo",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Support",
    ],
    alwayslink = True,
)

cc_library(
    name = "legalize_qdq_custom_call",
    srcs = ["transforms/torch/legalize_qdq_custom_call_pass.cc"],
    copts = ["-Ithird_party"],
    deps = [
        ":passes_inc_gen",
        ":prepare_hlo",
        "@com_google_absl//absl/log",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:AsmParser",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:QuantOps",
        "@toolchain-project//mlir:Support",
        "@stablehlo//:stablehlo_ops",
    ],
    alwayslink = True,
)

cc_library(
    name = "build_stablehlo_composite",
    srcs = ["transforms/torch/build_stablehlo_composite_pass.cc"],
    copts = ["-Ithird_party"],
    deps = [
        ":passes_inc_gen",
        ":prepare_hlo",
        "@com_google_absl//absl/strings",
        "@jsoncpp_git//:jsoncpp",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:Analysis",
        "@toolchain-project//mlir:ArithDialect",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:Support",
        "@stablehlo//:stablehlo_ops",
    ],
    alwayslink = True,
)

cc_library(
    name = "composite_lowering",
    srcs = [
        "transforms/composite_avg_pool.cc",
        "transforms/composite_lowering_pass.cc",
    ],
    hdrs = [
        "transforms/composite_avg_pool.h",
        "transforms/stablehlo_passes.h",
    ],
    copts = [
        "-Ithird_party",
    ],
    deps = [
        ":composite_lowering_inc_gen",
        ":composite_utils",
        ":passes_inc_gen",
        "//machina/compiler/mlir/lite:machina_lite",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:ArithDialect",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:Support",
        "@toolchain-project//mlir:TransformUtils",
        "@local_xla//xla/mlir_hlo",
    ],
    alwayslink = True,
)

cc_library(
    name = "optimize_layout",
    srcs = [
        "transforms/optimize_layout.cc",
    ],
    hdrs = ["transforms/stablehlo_passes.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":passes_inc_gen",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:DialectUtils",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:Support",
        "@toolchain-project//mlir:TransformUtils",
        "@stablehlo//:stablehlo_ops",
    ],
    alwayslink = 1,
)

gentbl_cc_library(
    name = "composite_lowering_inc_gen",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = {"transforms/generated_composite_lowering.inc": ["-gen-rewriters"]},
    tblgen = "@toolchain-project//mlir:mlir-tblgen",
    td_file = "transforms/composite_lowering_patterns.td",
    deps = [
        ":composite_td_files",
        ":composite_utils",
        "//machina/compiler/mlir/lite:machina_lite_ops_td_files",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:FuncTdFiles",
        "@local_xla//xla/mlir_hlo:hlo_ops_td_files",
    ],
)

tf_cc_binary(
    name = "odml_to_stablehlo",
    srcs = [
        "odml_to_stablehlo.cc",
    ],
    copts = ["-O3"],
    deprecation = "odml_to_stablehlo is being deprecated, please use TFlite converter with flag: " +
                  "converter.target_spec.supported_ops = " +
                  " [tf.lite.OpsSet.EXPERIMENTAL_STABLEHLO_OPS]",
    deps = [
        ":check_accepted_ops_pass",
        ":op_stat_pass",
        ":stablehlo_util",
        ":transforms",
        "//machina/cc/saved_model:loader",
        "//machina/compiler/mlir:init_mlir",
        "//machina/compiler/mlir:passes",
        "//machina/compiler/mlir/lite:canonicalize_boundary_value",
        "//machina/compiler/mlir/lite:flatbuffer_export",
        "//machina/compiler/mlir/lite:tf_to_tfl_flatbuffer",
        "//machina/compiler/mlir/quantization/machina:quantize_preprocess",
        "//machina/compiler/mlir/quantization/machina:tf_quant_ops",
        "//machina/compiler/mlir/stablehlo:legalize_tf",
        "//machina/compiler/mlir/machina",
        "//machina/compiler/mlir/machina:mlir_roundtrip_flags",
        "//machina/compiler/mlir/machina/transforms:machina_passes",
        "//machina/compiler/mlir/machina/transforms:tf_graph_optimization_pass",
        "//machina/compiler/mlir/tf2xla:compile_mlir_util",
        "//machina/compiler/mlir/tf2xla/transforms:xla_legalize_tf",
        "//machina/core:core_cpu_base",
        "//machina/core:lib",
        "//machina/core/ir/types:Dialect",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:AllPassesAndDialects",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Parser",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:QuantOps",
        "@toolchain-project//mlir:RegisterAllDialects",
        "@toolchain-project//mlir:RegisterAllPasses",
        "@toolchain-project//mlir:Support",
        "@local_xla//xla/mlir/framework/transforms:passes",
        "@local_xla//xla/mlir_hlo:all_passes",
        "@local_xla//xla/mlir_hlo:hlo_dialect_registration",
        "@stablehlo//:register",
    ],
)

tf_cc_binary(
    name = "odml-to-stablehlo-opt",
    testonly = True,
    tags = ["hostonly"],
    deps = [
        ":compose_uniform_quantized_type_pass",
        ":legalize_stablehlo_composite_to_tfl_custom",
        ":legalize_stablehlo_custom_call_to_composite",
        ":legalize_stablehlo_to_vhlo_pass",
        ":legalize_tf_xla_call_module_to_stablehlo_pass",
        ":optimize",
        ":passes_inc_gen",
        ":prepare_hlo",
        ":stablehlo_fuse_convolution_pass",
        ":stablehlo_unfuse_batch_norm_pass",
        ":tf_legalize_hlo",
        ":tfl_legalize_chlo",
        ":tfl_legalize_hlo",
        ":tfl_stablehlo",
        ":uniform_quantized_stablehlo_to_tfl_pass",
        "//machina/compiler/mlir:passes",
        "//machina/compiler/mlir/lite:litert_mlir_opt_main",
        "//machina/compiler/mlir/stablehlo:fold_broadcast_pass",
        "//machina/compiler/mlir/stablehlo:fuse_convolution_pass",
        "//machina/compiler/mlir/stablehlo:tf_stablehlo",
        "//machina/compiler/mlir/stablehlo:unfuse_batch_norm_pass",
    ],
)

exports_files([
    "run_lit.sh",
])

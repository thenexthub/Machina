# TensorFlow -> TOSA Compiler Bridge.
# See:
#   https://developer.mlplatform.org/w/tosa/
#   https://github.com/thenexthub/Codira/blob/main/mlir/docs/Dialects/TOSA.md

load("@toolchain-project//mlir:tblgen.bzl", "gentbl_cc_library")
load("//machina:machina.bzl", "tf_cc_binary")
load("//machina:machina.default.bzl", "get_compatible_with_portable")

# TODO: Tighten visibility once targets are at the right granularity.
package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = [":internal"],
    licenses = ["notice"],
)

package_group(
    name = "internal",
    packages = [
        "//machina/compiler/mlir/tosa/...",
    ],
)

filegroup(
    name = "tosa_ops_td_files",
    srcs = [
        "@toolchain-project//mlir:TosaDialectTdFiles",
    ],
    compatible_with = get_compatible_with_portable(),
)

gentbl_cc_library(
    name = "tosa_passes_inc_gen",
    compatible_with = get_compatible_with_portable(),
    tags = ["tf_tosa"],
    tbl_outs = [
        (
            [
                "-gen-pass-decls",
                "-name=LegalizeTosa",
            ],
            "transforms/passes.h.inc",
        ),
    ],
    tblgen = "@toolchain-project//mlir:mlir-tblgen",
    td_file = "transforms/passes.td",
    deps = [
        "@toolchain-project//mlir:PassBaseTdFiles",
    ],
)

cc_library(
    name = "passes_header",
    hdrs = [
        "transforms/passes.h",
        "transforms/passes.h.inc",
    ],
    compatible_with = get_compatible_with_portable(),
    tags = ["tf_tosa"],
    deps = [
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Pass",
    ],
)

cc_library(
    name = "legalize_common",
    srcs = [
        "transforms/legalize_common.cc",
        "transforms/legalize_utils.cc",
    ],
    hdrs = [
        "transforms/legalize_common.h",
        "transforms/legalize_utils.h",
    ],
    compatible_with = get_compatible_with_portable(),
    tags = ["tf_tosa"],
    deps = [
        "//machina/compiler/mlir/lite:machina_lite",
        "//machina/compiler/mlir/lite/kernels/internal:common",
        "//machina/compiler/mlir/lite/kernels/internal:quantization_util",
        "//machina/compiler/mlir/lite/quantization/ir:QuantOps",
        "//machina/compiler/mlir/machina:dynamic_shape_utils",
        "//machina/core:framework",
        "//machina/core/kernels:conv_grad_shape_utils",
        "//machina/lite/kernels/internal:reference_base",
        "@com_google_absl//absl/status",
        "@gemmlowp",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:ArithDialect",
        "@toolchain-project//mlir:ArithUtils",
        "@toolchain-project//mlir:DialectUtils",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:InferTypeOpInterface",
        "@toolchain-project//mlir:QuantOps",
        "@toolchain-project//mlir:Rewrite",
        "@toolchain-project//mlir:Support",
        "@toolchain-project//mlir:TensorDialect",
        "@toolchain-project//mlir:TosaDialect",
        "@toolchain-project//mlir:TransformUtils",
        "@toolchain-project//mlir:Transforms",
        "@local_xla//xla/tsl/framework/fixedpoint",
    ],
)

gentbl_cc_library(
    name = "tosa_legalize_tf_inc_gen",
    compatible_with = get_compatible_with_portable(),
    tags = ["tf_tosa"],
    tbl_outs = [
        (
            ["-gen-rewriters"],
            "transforms/tf_legalize_patterns.inc",
        ),
    ],
    tblgen = "@toolchain-project//mlir:mlir-tblgen",
    td_file = "transforms/tf_legalize_patterns.td",
    deps = [
        ":tosa_ops_td_files",
        "//machina/compiler/mlir/lite/quantization/ir:QuantizationOpsTdFiles",
        "//machina/compiler/mlir/machina:machina_ops_td_files",
        "@toolchain-project//mlir:FuncTdFiles",
    ],
)

cc_library(
    name = "tf_passes",
    srcs = [
        "tf_passes.cc",
        "transforms/fuse_bias_tf.cc",
        "transforms/legalize_tf.cc",
        "transforms/tf_legalize_patterns.inc",
    ],
    hdrs = [
        "tf_passes.h",
        "transforms/passes.h",
    ],
    compatible_with = get_compatible_with_portable(),
    tags = ["tf_tosa"],
    deps = [
        ":legalize_common",
        ":passes_header",
        ":tosa_legalize_tf_inc_gen",
        "//machina/compiler/mlir/lite:machina_lite",
        "//machina/compiler/mlir/lite/quantization/ir:QuantOps",
        "//machina/compiler/mlir/machina",
        "//machina/compiler/mlir/machina:dynamic_shape_utils",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:AffineTransforms",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:QuantOps",
        "@toolchain-project//mlir:Support",
        "@toolchain-project//mlir:TosaDialect",
        "@toolchain-project//mlir:TransformUtils",
        "@toolchain-project//mlir:Transforms",
    ],
)

gentbl_cc_library(
    name = "tosa_legalize_tfl_inc_gen",
    compatible_with = get_compatible_with_portable(),
    tags = ["tf_tosa"],
    tbl_outs = [
        (
            ["-gen-rewriters"],
            "transforms/tfl_legalize_patterns.inc",
        ),
    ],
    tblgen = "@toolchain-project//mlir:mlir-tblgen",
    td_file = "transforms/tfl_legalize_patterns.td",
    deps = [
        ":tosa_ops_td_files",
        "//machina/compiler/mlir/lite:machina_lite_ops_td_files",
        "//machina/compiler/mlir/lite/quantization/ir:QuantizationOpsTdFiles",
        "@toolchain-project//mlir:FuncTdFiles",
    ],
)

cc_library(
    name = "tfl_passes",
    srcs = [
        "tfl_passes.cc",
        "transforms/convert_metadata.cc",
        "transforms/convert_tfl_uint8.cc",
        "transforms/dequantize_tfl_softmax.cc",
        "transforms/legalize_tfl.cc",
        "transforms/legalize_tfl_stateful.cc",
        "transforms/lower_complex_types.cc",
        "transforms/retain_call_once_funcs.cc",
        "transforms/strip_metadata.cc",
        "transforms/strip_quant_types.cc",
        "transforms/tfl_legalize_patterns.inc",
        "transforms/verify_fully_converted.cc",
    ],
    hdrs = [
        "tfl_passes.h",
        "transforms/passes.h",
    ],
    compatible_with = get_compatible_with_portable(),
    tags = ["tf_tosa"],
    deps = [
        ":legalize_common",
        ":passes_header",
        ":tosa_legalize_tfl_inc_gen",
        "//machina/compiler/mlir/lite:machina_lite",
        "//machina/compiler/mlir/lite:machina_lite_quantize",
        "//machina/compiler/mlir/lite/quantization/ir:QuantOps",
        "//machina/compiler/mlir/machina:dynamic_shape_utils",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:AffineTransforms",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:FunctionInterfaces",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:QuantOps",
        "@toolchain-project//mlir:ReconcileUnrealizedCasts",
        "@toolchain-project//mlir:Support",
        "@toolchain-project//mlir:TosaDialect",
        "@toolchain-project//mlir:TransformUtils",
        "@toolchain-project//mlir:Transforms",
    ],
)

cc_library(
    name = "tf_tfl_passes",
    srcs = [
        "tf_tfl_passes.cc",
        "transforms/legalize_tf_tfl.cc",
    ],
    hdrs = [
        "tf_tfl_passes.h",
        "transforms/passes.h",
    ],
    compatible_with = get_compatible_with_portable(),
    tags = ["tf_tosa"],
    deps = [
        ":legalize_common",
        ":passes_header",
        ":tf_passes",
        ":tfl_passes",
        "//machina/compiler/mlir/lite:lift_tflite_flex_ops",
        "//machina/compiler/mlir/lite:machina_lite",
        "@toolchain-project//mlir:AffineTransforms",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:Support",
        "@toolchain-project//mlir:TosaDialect",
        "@toolchain-project//mlir:TransformUtils",
        "@toolchain-project//mlir:Transforms",
    ],
)

tf_cc_binary(
    name = "tf-tosa-opt",
    testonly = True,
    srcs = ["tf_tosa_opt.cc"],
    tags = ["tf_tosa"],
    deps = [
        "//machina/compiler/mlir:init_mlir",
        "//machina/compiler/mlir:passes",
        "//machina/compiler/mlir:register_common_dialects",
        "//machina/compiler/mlir/lite:machina_lite",
        "//machina/compiler/mlir/lite:tf_tfl_passes",  # buildcleaner:keep
        "//machina/compiler/mlir/quantization/stablehlo:bridge_passes",
        "//machina/compiler/mlir/machina:mlprogram_util",
        "//machina/compiler/mlir/machina/transforms:machina_passes",
        "//machina/compiler/mlir/machina/transforms:machina_test_passes",
        "//machina/compiler/mlir/machina/transforms:tf_graph_optimization_pass",
        "//machina/compiler/mlir/machina/transforms:tf_saved_model_passes",  # buildcleaner:keep
        "//machina/compiler/mlir/machina/transforms/host_runtime:lower_cluster_to_runtime_ops",
        "//machina/compiler/mlir/machina/transforms/host_runtime:runtime_passes",
        "//machina/compiler/mlir/machina/transforms/sparsecore:sparsecore_passes",
        "//machina/compiler/mlir/tf2xla:compile_mlir_util",
        "//machina/compiler/mlir/tf2xla/internal/passes:clustering_passes",
        "//machina/compiler/mlir/tf2xla/internal/passes:mlir_to_graph_passes",
        "//machina/compiler/mlir/tf2xla/transforms:tf_xla_passes",
        "//machina/compiler/mlir/tf2xla/transforms:xla_legalize_tf",
        "//machina/compiler/mlir/tosa:tf_passes",
        "//machina/compiler/mlir/tosa:tf_tfl_passes",
        "//machina/compiler/mlir/tosa:tfl_passes",
        "@toolchain-project//mlir:AllPassesAndDialects",
        "@toolchain-project//mlir:MlirOptLib",
        "@toolchain-project//mlir:Support",
        "@toolchain-project//mlir:Transforms",
        "@local_xla//xla/mlir/framework/ir:xla_framework",
        "@local_xla//xla/mlir/framework/transforms:passes",
        "@local_xla//xla/mlir_hlo:all_passes",
    ],
)

filegroup(
    name = "litfiles",
    srcs = glob(["runlit*py"]),
    visibility = ["//machina:__subpackages__"],
)

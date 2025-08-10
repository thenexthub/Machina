load("@toolchain-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//machina:strict.default.bzl", "py_strict_binary")

# Placeholder: load py_proto_library
load("//machina:machina.bzl", "tf_cc_binary")
load("//machina:machina.default.bzl", "get_compatible_with_portable")
load("//machina/compiler/mlir/quantization/machina:internal_visibility_allowlist.bzl", "internal_visibility_allowlist")
load("//machina/core/platform:build_config.bzl", "tf_proto_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = [
        ":internal_visibility_allowlist_package",
        "//machina:__pkg__",
    ],
    licenses = ["notice"],
)

# TODO(b/249181641): Restructure source codes for TF quantization.
package_group(
    name = "internal_visibility_allowlist_package",
    packages = [
        "//machina/compiler/mlir/lite/...",
        "//machina/compiler/mlir/quantization/...",
        "//machina/compiler/mlir/machina_to_stablehlo/...",
    ] + internal_visibility_allowlist(),
)

py_strict_binary(
    name = "gen_quantized_function_library",
    srcs = ["gen_quantized_function_library.py"],
    deps = [
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
    ],
)

genrule(
    name = "quantized_function_library",
    srcs = [
        "passes/quantized_function_library_uniform_quantized.mlir",
        "passes/quantized_function_library.mlir",
        "passes/quantized_function_library_uniform_quantized_drq.mlir",
        "passes/quantized_function_library_tf_drq.mlir",
        "passes/quantized_function_library_xla_weight_only.mlir",
    ],
    outs = [
        "passes/quantized_function_library.h",
    ],
    cmd = "$(location gen_quantized_function_library) --output_file $(RULEDIR)/passes/quantized_function_library.h --src '$(SRCS)'",
    compatible_with = get_compatible_with_portable(),
    tools = ["gen_quantized_function_library"],
)

cc_library(
    name = "manipulate_model_attr",
    srcs = [
        "passes/manipulate_model_attr.cc",
    ],
    hdrs = [
        "passes/manipulate_model_attr.h",
    ],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//machina/compiler/mlir/machina",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
    ],
)

cc_library(
    name = "remove_identity_op_pattern",
    srcs = [
        "passes/remove_identity_op_pattern.cc",
    ],
    hdrs = [
        "passes/remove_identity_op_pattern.h",
    ],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//machina/compiler/mlir/machina",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Support",
        "@toolchain-project//mlir:TransformUtils",
    ],
)

td_library(
    name = "quant_td_files",
    srcs = [
        "passes/cast_bf16_ops_to_f32.td",
        "passes/lift_quantizable_spots_as_functions.td",
        "passes/lift_quantizable_spots_as_functions_drq.td",
        "passes/optimize.td",
        "passes/post_quantize.td",
        "passes/prepare_lifting.td",
        "passes/prepare_quantize.td",
        "passes/preprocess_op.td",
        "passes/quantize_composite_functions.td",
        "passes/replace_cast_hacks_with_tf_xla_ops.td",
        "passes/tf_quant_ops.td",
    ],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//machina/compiler/mlir/quantization/common:quant_td_files",
        "//machina/compiler/mlir/quantization/common/ir:QuantizationOpsTdFiles",
        "//machina/compiler/mlir/machina:machina_ops_td_files",
        "@toolchain-project//mlir:ArithOpsTdFiles",
        "@toolchain-project//mlir:FuncTdFiles",
    ],
)

gentbl_cc_library(
    name = "convert_tf_xla_op_to_tf_op_inc_gen",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = {"passes/convert_tf_xla_op_to_tf_op.inc": ["-gen-rewriters"]},
    tblgen = "@toolchain-project//mlir:mlir-tblgen",
    td_file = "passes/convert_tf_xla_op_to_tf_op.td",
    deps = [":quant_td_files"],
)

gentbl_cc_library(
    name = "cast_bf16_ops_to_f32_inc_gen",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = {"passes/cast_bf16_ops_to_f32.inc": ["-gen-rewriters"]},
    tblgen = "@toolchain-project//mlir:mlir-tblgen",
    td_file = "passes/cast_bf16_ops_to_f32.td",
    deps = [":quant_td_files"],
)

gentbl_cc_library(
    name = "prepare_lifting_inc_gen",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = {"passes/prepare_lifting.inc": ["-gen-rewriters"]},
    tblgen = "@toolchain-project//mlir:mlir-tblgen",
    td_file = "passes/prepare_lifting.td",
    deps = [":quant_td_files"],
)

gentbl_cc_library(
    name = "lift_quantizable_spots_as_functions_inc_gen",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = {"passes/lift_quantizable_spots_as_functions.inc": ["-gen-rewriters"]},
    tblgen = "@toolchain-project//mlir:mlir-tblgen",
    td_file = "passes/lift_quantizable_spots_as_functions.td",
    deps = [":quant_td_files"],
)

gentbl_cc_library(
    name = "lift_quantizable_spots_as_functions_drq_inc_gen",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = {"passes/lift_quantizable_spots_as_functions_drq.inc": ["-gen-rewriters"]},
    tblgen = "@toolchain-project//mlir:mlir-tblgen",
    td_file = "passes/lift_quantizable_spots_as_functions_drq.td",
    deps = [":quant_td_files"],
)

gentbl_cc_library(
    name = "prepare_quantize_inc_gen",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = {"passes/prepare_quantize.inc": ["-gen-rewriters"]},
    tblgen = "@toolchain-project//mlir:mlir-tblgen",
    td_file = "passes/prepare_quantize.td",
    deps = [":quant_td_files"],
)

gentbl_cc_library(
    name = "quantize_composite_functions_inc_gen",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = {"passes/quantize_composite_functions.inc": ["-gen-rewriters"]},
    tblgen = "@toolchain-project//mlir:mlir-tblgen",
    td_file = "passes/quantize_composite_functions.td",
    deps = [":quant_td_files"],
)

gentbl_cc_library(
    name = "tf_quant_ops_inc_gen",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = {
        "passes/tf_quant_ops.h.inc": ["-gen-op-decls"],
        "passes/tf_quant_ops.cc.inc": ["-gen-op-defs"],
    },
    tblgen = "@toolchain-project//mlir:mlir-tblgen",
    td_file = "passes/tf_quant_ops.td",
    deps = [
        ":quant_td_files",
    ],
)

gentbl_cc_library(
    name = "optimize_inc_gen",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = {"passes/optimize.inc": ["-gen-rewriters"]},
    tblgen = "@toolchain-project//mlir:mlir-tblgen",
    td_file = "passes/optimize.td",
    deps = [":quant_td_files"],
)

gentbl_cc_library(
    name = "convert_tpu_model_to_cpu_inc_gen",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = {"passes/convert_tpu_model_to_cpu.inc": ["-gen-rewriters"]},
    tblgen = "@toolchain-project//mlir:mlir-tblgen",
    td_file = "passes/convert_tpu_model_to_cpu.td",
    deps = [":quant_td_files"],
)

gentbl_cc_library(
    name = "post_quantize_inc_gen",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = {"passes/post_quantize.inc": ["-gen-rewriters"]},
    tblgen = "@toolchain-project//mlir:mlir-tblgen",
    td_file = "passes/post_quantize.td",
    deps = [":quant_td_files"],
)

gentbl_cc_library(
    name = "preprocess_op_gen",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = {"passes/preprocess_op.inc": ["-gen-rewriters"]},
    tblgen = "@toolchain-project//mlir:mlir-tblgen",
    td_file = "passes/preprocess_op.td",
    deps = [":quant_td_files"],
)

cc_library(
    name = "tf_quant_ops",
    srcs = [
        "passes/tf_quant_ops.cc",
        "passes/tf_quant_ops.cc.inc",
        "passes/tf_quant_ops.h.inc",
    ],
    hdrs = ["passes/tf_quant_ops.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = ["//visibility:public"],
    deps = [
        "//machina/compiler/mlir/machina:machina_attributes",
        "//machina/compiler/mlir/machina:machina_op_interfaces",
        "//machina/compiler/mlir/machina:machina_side_effects",
        "//machina/compiler/mlir/machina:machina_structs",
        "//machina/compiler/mlir/machina:machina_traits",
        "//machina/compiler/mlir/machina:machina_types",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:CallOpInterfaces",
        "@toolchain-project//mlir:ControlFlowInterfaces",
        "@toolchain-project//mlir:DerivedAttributeOpInterface",
        "@toolchain-project//mlir:Dialect",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:InferTypeOpInterface",
        "@toolchain-project//mlir:LoopLikeInterface",
        "@toolchain-project//mlir:Parser",
        "@toolchain-project//mlir:SideEffectInterfaces",
        "@toolchain-project//mlir:Support",
    ],
)

gentbl_cc_library(
    name = "replace_cast_hacks_with_tf_xla_ops_inc_gen",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = {"passes/replace_cast_hacks_with_tf_xla_ops.inc": ["-gen-rewriters"]},
    tblgen = "@toolchain-project//mlir:mlir-tblgen",
    td_file = "passes/replace_cast_hacks_with_tf_xla_ops.td",
    deps = [":quant_td_files"],
)

cc_library(
    name = "passes",
    srcs = [
        "passes/add_dump_tensor_op.cc",
        "passes/add_quantization_unit_loc.cc",
        "passes/cast_bf16_ops_to_f32.cc",
        "passes/cast_bf16_ops_to_f32.inc",
        "passes/convert_custom_aggregation_op_to_quant_stats.cc",
        "passes/convert_fake_quant_to_qdq.cc",
        "passes/convert_tf_xla_op_to_tf_op.cc",
        "passes/convert_tf_xla_op_to_tf_op.inc",
        "passes/convert_tpu_model_to_cpu.cc",
        "passes/convert_tpu_model_to_cpu.inc",
        "passes/duplicate_shape_determining_constants.cc",
        "passes/insert_custom_aggregation_ops.cc",
        "passes/insert_main_function.cc",
        "passes/insert_quantized_functions.cc",
        "passes/insert_restore_op.cc",
        "passes/insert_save_op.cc",
        "passes/lift_hashtable_ops_as_args.cc",
        "passes/lift_quantizable_spots_as_functions.cc",
        "passes/lift_quantizable_spots_as_functions.inc",
        "passes/lift_quantizable_spots_as_functions_drq.cc",
        "passes/lift_quantizable_spots_as_functions_drq.inc",
        "passes/mark_functions_noinline.cc",
        "passes/merge_duplicate_resource_ops.cc",
        "passes/merge_initializer_function_ops_to_main.cc",
        "passes/merge_save_function_ops_to_main.cc",
        "passes/optimize.cc",
        "passes/optimize.inc",
        "passes/post_quantize.cc",
        "passes/post_quantize.inc",
        "passes/prepare_lifting.cc",
        "passes/prepare_lifting.inc",
        "passes/prepare_quantize.cc",
        "passes/prepare_quantize.inc",
        "passes/prepare_quantize_drq.cc",
        "passes/preprocess_op.cc",
        "passes/propagate_quantize_type.cc",
        "passes/quantize.cc",
        "passes/quantize_composite_functions.cc",
        "passes/quantize_composite_functions.inc",
        "passes/quantize_weights.cc",
        "passes/quantized_function_library.h",
        "passes/remove_var_init_by_const.cc",
        "passes/replace_cast_hacks_with_tf_xla_ops.cc",
        "passes/replace_cast_hacks_with_tf_xla_ops.inc",
        "passes/unfreeze_constants.cc",
    ],
    hdrs = [
        "passes/constants.h",
        "passes/passes.h",
    ],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":cast_bf16_ops_to_f32_inc_gen",
        ":convert_tf_xla_op_to_tf_op_inc_gen",
        ":convert_tpu_model_to_cpu_inc_gen",
        ":lift_quantizable_spots_as_functions_drq_inc_gen",
        ":lift_quantizable_spots_as_functions_inc_gen",
        ":manipulate_model_attr",
        ":optimize_inc_gen",
        ":post_quantize_inc_gen",
        ":prepare_lifting_inc_gen",
        ":prepare_quantize_inc_gen",
        ":preprocess_op_gen",
        ":quantization_options_proto_cc",
        ":quantize_composite_functions_inc_gen",
        ":remove_identity_op_pattern",
        ":replace_cast_hacks_with_tf_xla_ops_inc_gen",
        ":tf_quant_ops",
        "//machina/compiler/mlir/quantization/common:attrs_and_constraints",
        "//machina/compiler/mlir/quantization/common:func",
        "//machina/compiler/mlir/quantization/common:lift_as_function_call",
        "//machina/compiler/mlir/quantization/common/ir:QuantOps",
        "//machina/compiler/mlir/quantization/common/quantization_lib",
        "//machina/compiler/mlir/quantization/common/quantization_lib:quantization_config",
        "//machina/compiler/mlir/quantization/stablehlo:quantization_config_proto_cc",
        "//machina/compiler/mlir/quantization/stablehlo/cc/calibration:calibration_parameters",
        "//machina/compiler/mlir/quantization/machina/cc:const_op_size",
        "//machina/compiler/mlir/quantization/machina/cc:constant_fold",
        "//machina/compiler/mlir/quantization/machina/cc:quantization_unit_loc",
        "//machina/compiler/mlir/quantization/machina/cc:run_passes",
        "//machina/compiler/mlir/quantization/machina/ops:tf_op_quant_spec",
        "//machina/compiler/mlir/quantization/machina/ops:tf_quantize_op",
        "//machina/compiler/mlir/quantization/machina/utils:fake_quant_utils",
        "//machina/compiler/mlir/quantization/machina/utils:tf_to_uniform_attribute_utils",
        "//machina/compiler/mlir/quantization/machina/utils:tf_to_xla_attribute_utils",
        "//machina/compiler/mlir/machina",
        "//machina/compiler/mlir/machina:error_util",
        "//machina/compiler/mlir/machina:import_model",
        "//machina/compiler/mlir/machina:mangling_util",
        "//machina/compiler/mlir/machina:machina_attributes",
        "//machina/compiler/mlir/machina:machina_types",
        "//machina/compiler/mlir/machina:xla_call_module_attrs",
        "//machina/compiler/mlir/machina/transforms:machina_passes",
        "//machina/compiler/mlir/machina/transforms:tf_dialect_passes",
        "//machina/compiler/mlir/machina/transforms:tf_saved_model_freeze_variables",
        "//machina/compiler/mlir/machina/transforms:tf_saved_model_passes",
        "//machina/compiler/mlir/utils:name_utils",
        "//machina/core:protos_all_cc",
        "//machina/core/ir/importexport:convert_tensor",
        "//machina/core/platform:macros",
        "//machina/core/platform:path",
        "//machina/core/tpu:tpu_defs",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_googlesource_code_re2//:re2",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:ArithDialect",
        "@toolchain-project//mlir:Dialect",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:FunctionInterfaces",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Parser",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:QuantOps",
        "@toolchain-project//mlir:Rewrite",
        "@toolchain-project//mlir:Support",
        "@toolchain-project//mlir:TransformUtils",
        "@toolchain-project//mlir:Transforms",
        "@toolchain-project//mlir:UBDialect",
        "@local_xla//xla:xla_data_proto_cc",
    ],
    # Alwayslink is required for registering the MLIR passes.
    # TODO(b/255530126): Split the pass registration from the definitions to avoid binary size bloat.
    alwayslink = True,
)

cc_library(
    name = "quantize_preprocess",
    srcs = [
        "quantize_preprocess.cc",
    ],
    hdrs = [
        "quantize_preprocess.h",
    ],
    compatible_with = get_compatible_with_portable(),
    visibility = ["//visibility:public"],
    deps = [
        ":passes",
        "//machina/compiler/mlir/quantization/stablehlo:bridge_passes",
        "//machina/compiler/mlir/quantization/stablehlo/cc:pass_pipeline",
        "//machina/compiler/mlir/quantization/machina/cc:run_passes",
        "//machina/compiler/mlir/stablehlo:fold_broadcast_pass",
        "//machina/compiler/mlir/stablehlo:fuse_convolution_pass",
        "//machina/compiler/mlir/stablehlo:legalize_tf_xla_call_module_to_stablehlo_pass",
        "//machina/compiler/mlir/stablehlo:rename_entrypoint_to_main",
        "//machina/compiler/mlir/stablehlo:tf_stablehlo",
        "//machina/compiler/mlir/stablehlo:unfuse_batch_norm_pass",
        "//machina/compiler/mlir/machina:error_util",
        "//machina/compiler/mlir/machina/transforms:machina_passes",
        "//machina/compiler/mlir/machina/transforms:tf_dialect_passes",
        "//machina/compiler/mlir/machina/transforms:tf_saved_model_freeze_variables",
        "//machina/compiler/mlir/machina/transforms:tf_saved_model_passes",
        "//machina/core:core_cpu_base",
        "//machina/core/platform:path",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:Transforms",
        "@local_xla//xla/mlir_hlo:all_passes",
    ],
)

cc_library(
    name = "quantize_passes",
    srcs = ["quantize_passes.cc"],
    hdrs = ["quantize_passes.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = ["//visibility:public"],
    deps = [
        ":passes",
        ":quantization_options_proto_cc",
        "//machina/compiler/mlir/machina/transforms:machina_passes",
        "//machina/compiler/mlir/machina/transforms:tf_dialect_passes",
        "//machina/core/platform:path",
        "@com_google_absl//absl/strings",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:Transforms",
        "@local_xla//xla/mlir_hlo:mhlo_passes",
    ],
)

# OSS only: This target is header-only. Link `quantization_options_proto_impl` only to
# `libmachina_framework.so` via `lib_internal_impl`. Do NOT link
# `quantization_options_proto_impl` directly unless the target does not link
# `libmachina_framework.so`.
tf_proto_library(
    name = "quantization_options_proto",
    srcs = [
        "quantization_options.proto",
    ],
    make_default_target_header_only = True,
    protodeps = [
        "//machina/core:protos_all",
        "//machina/compiler/mlir/quantization/stablehlo:quantization_config_proto",
    ],
    visibility = ["//visibility:public"],
)

# copybara:uncomment_begin(google-only)
# py_proto_library(
#     name = "quantization_options_py_pb2",
#     visibility = ["//visibility:public"],
#     deps = [":quantization_options_proto"],
# )
# copybara:uncomment_end

# OSS only: This target is header-only. Link `exported_model_proto_impl` only to
# `libmachina_framework.so` via `lib_internal_impl`. Do NOT link
# `exported_model_proto_impl` directly unless the target does not link
# `libmachina_framework.so`.
tf_proto_library(
    name = "exported_model_proto",
    srcs = ["exported_model.proto"],
    make_default_target_header_only = True,
    protodeps = [
        "//machina/core:protos_all",
    ],
    visibility = [
        ":internal_visibility_allowlist_package",
        # To be visible from `lib_internal_impl`.
        "//machina/core:__pkg__",
        "//machina/python:__pkg__",
    ],
)

# copybara:uncomment_begin(google-only)
# py_proto_library(
#     name = "exported_model_py_pb2",
#     deps = [":exported_model_proto"],
# )
# copybara:uncomment_end

tf_cc_binary(
    name = "tf-quant-opt",
    srcs = ["passes/tf_quant_opt.cc"],
    deps = [
        ":passes",
        "//machina/compiler/mlir:init_mlir",
        "//machina/compiler/mlir/quantization/common/ir:QuantOps",
        "//machina/compiler/mlir/machina",
        "//machina/compiler/mlir/machina/transforms:machina_passes",
        "//machina/compiler/mlir/machina/transforms:tf_dialect_passes",
        "//machina/core/ir/types:Dialect",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:AllPassesAndDialects",
        "@toolchain-project//mlir:ArithDialect",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:FuncExtensions",
        "@toolchain-project//mlir:MlirOptLib",
        "@toolchain-project//mlir:QuantOps",
        "@toolchain-project//mlir:RegisterAllPasses",
        "@toolchain-project//mlir:SCFDialect",
        "@toolchain-project//mlir:ShapeDialect",
        "@stablehlo//:stablehlo_ops",
    ],
)

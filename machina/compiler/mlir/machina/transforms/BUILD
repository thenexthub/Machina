load("@toolchain-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")
load("//machina:machina.bzl", "tf_cc_test")
load("//machina:machina.default.bzl", "filegroup", "get_compatible_with_portable")
load("//machina/core/platform:rules_cc.bzl", "cc_library")

# copybara:uncomment_end(google-only)
package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

gentbl_cc_library(
    name = "machina_canonicalize_inc_gen",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = {"generated_canonicalize.inc": ["-gen-rewriters"]},
    tblgen = "@toolchain-project//mlir:mlir-tblgen",
    td_file = "canonicalize.td",
    deps = [
        ":rewrite_util_td_files",
        "//machina/compiler/mlir/machina:machina_ops_td_files",
    ],
)

gentbl_cc_library(
    name = "machina_reduce_patterns_inc_gen",
    tbl_outs = {"reducer/tf_reduce_patterns.inc": ["-gen-rewriters"]},
    tblgen = "@toolchain-project//mlir:mlir-tblgen",
    td_file = "reducer/tf_mlir_reduce_patterns.td",
    deps = [
        "//machina/compiler/mlir/machina:machina_ops_td_files",
    ],
)

cc_library(
    name = "tfe_legalize_tfg",
    srcs = [
        "passes.h",
        "tfg-to-tfe.cc",
    ],
    deps = [
        ":tf_device_pass_inc_gen",
        ":tf_pass_inc_gen",
        "//machina/compiler/mlir/machina",
        "//machina/core:framework",
        "//machina/core/transforms/toposort:Pass",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:Support",
        "@toolchain-project//mlir:TransformUtils",
        "@shardy//shardy/dialect/sdy/ir:dialect",
    ],
)

cc_library(
    name = "mlprogram",
    srcs = [
        "mlprogram.cc",
    ],
    hdrs = [
        "mlprogram.h",
    ],
    deps = [
        ":bridge",
        ":machina_passes",
        ":tf_saved_model_passes",
        "//machina/compiler/mlir/tf2xla/internal:clustering_bridge_passes",
        "//machina/compiler/mlir/tf2xla/transforms:legalize_tf",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:Support",
        "@toolchain-project//mlir:Transforms",
        "@local_xla//xla/mlir_hlo:mhlo_passes",
    ],
)

gentbl_cc_library(
    name = "decompose_resource_ops_inc_gen",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = {"generated_decompose_resource_ops.inc": ["-gen-rewriters"]},
    tblgen = "@toolchain-project//mlir:mlir-tblgen",
    td_file = "decompose_resource_ops.td",
    deps = [
        ":rewrite_util_td_files",
        "//machina/compiler/mlir/machina:machina_ops_td_files",
        "@toolchain-project//mlir:FuncTdFiles",
    ],
)

cc_library(
    name = "decompose_resource_ops",
    srcs = [
        "decompose_resource_ops.cc",
    ],
    hdrs = [
        "decompose_resource_ops.h",
    ],
    deps = [
        ":decompose_resource_ops_inc_gen",
        ":rewrite_util",
        "//machina/compiler/mlir/machina",
        "//machina/compiler/mlir/machina:machina_types",
        "//machina/core:framework",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:IR",
    ],
)

td_library(
    name = "rewrite_util_td_files",
    srcs = [
        "rewrite_util.td",
    ],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "@toolchain-project//mlir:OpBaseTdFiles",
    ],
)

cc_library(
    name = "rewrite_util",
    srcs = [
        "rewrite_util.cc",
    ],
    hdrs = [
        "rewrite_util.h",
    ],
    deps = [
        "//machina/compiler/mlir/machina:attribute_utils",
        "//machina/core:framework",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Support",
    ],
)

gentbl_cc_library(
    name = "tf_data_optimization_inc_gen",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = {"generated_tf_data_optimization.inc": ["-gen-rewriters"]},
    tblgen = "@toolchain-project//mlir:mlir-tblgen",
    td_file = "tf_data_optimization.td",
    deps = [
        "//machina/compiler/mlir/machina:machina_ops_td_files",
        "@toolchain-project//mlir:FuncTdFiles",
    ],
)

cc_library(
    name = "tf_data_optimization",
    srcs = [
        "tf_data_optimization.cc",
    ],
    hdrs = [
        "tf_data_optimization.h",
    ],
    deps = [
        ":tf_data_optimization_inc_gen",
        "//machina/compiler/mlir/machina",
        "//machina/compiler/mlir/machina:machina_types",
        "@toolchain-project//mlir:IR",
    ],
)

cc_library(
    name = "unroll_batch_matmul_pass",
    srcs = [
        "unroll_batch_matmul.cc",
    ],
    hdrs = [
        "unroll_batch_matmul.h",
    ],
    deps = [
        ":tf_pass_inc_gen",
        "//machina/compiler/mlir/machina",
        "//machina/core:framework",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/memory",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:AffineAnalysis",
        "@toolchain-project//mlir:Analysis",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:Support",
        "@toolchain-project//mlir:TransformUtils",
    ],
)

cc_library(
    name = "lift_variables_lib",
    srcs = [
        "lift_variables.cc",
    ],
    hdrs = [
        "lift_variables.h",
    ],
    deps = [
        "//machina/compiler/mlir/machina",
        "//machina/compiler/mlir/machina:convert_tensor",
        "//machina/core:core_cpu",
        "//machina/core:core_cpu_lib",
        "//machina/core:framework",
        "//machina/core:framework_internal",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core/platform:errors",
        "//machina/core/platform:threadpool_options",
        "@com_google_absl//absl/status:statusor",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Support",
    ],
)

cc_library(
    name = "mark_initialized_variables_lib",
    srcs = [
        "mark_initialized_variables.cc",
    ],
    hdrs = [
        "mark_initialized_variables.h",
    ],
    deps = [
        "//machina/compiler/mlir/machina:session_utils",
        "//machina/compiler/mlir/machina:machina_ops",
        "//machina/compiler/mlir/utils:string_container_utils",
        "//machina/core:core_cpu_base",
        "//machina/core:framework",
        "//machina/core:framework_internal",
        "//machina/core:protos_all_cc",
        "@com_google_absl//absl/strings",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Pass",
    ],
)

cc_library(
    name = "tf_saved_model_freeze_variables",
    srcs = [
        "tf_saved_model_freeze_variables.cc",
    ],
    hdrs = [
        "tf_saved_model_freeze_variables.h",
    ],
    deps = [
        ":tf_saved_model_freeze_utils",
        "//machina/compiler/mlir/machina",
        "//machina/compiler/mlir/machina:convert_tensor",
        "//machina/compiler/mlir/machina:resource_value_typed_analyzer",
        "//machina/compiler/mlir/machina:session_utils",
        "//machina/core:core_cpu_base",
        "//machina/core:framework_internal",
        "//machina/core:protos_all_cc",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Support",
    ],
)

cc_library(
    name = "tf_saved_model_freeze_utils",
    srcs = [
        "tf_saved_model_freeze_utils.cc",
    ],
    hdrs = [
        "tf_saved_model_freeze_utils.h",
    ],
    deps = [
        "//machina/compiler/mlir/machina",
        "//machina/core:framework_internal",
        "//machina/core:protos_all_cc",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Support",
    ],
)

cc_library(
    name = "initialize_variables_in_session_init",
    srcs = [
        "initialize_variables_in_session_init.cc",
    ],
    hdrs = [
        "initialize_variables_in_session_init.h",
    ],
    deps = [
        "//machina/compiler/mlir/machina",
        "//machina/compiler/mlir/machina:convert_tensor",
        "//machina/compiler/mlir/machina:session_utils",
        "//machina/compiler/mlir/machina:machina_ops",
        "//machina/core:core_cpu_base",
        "//machina/core:framework_internal",
        "//machina/core:protos_all_cc",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:ArithDialect",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:Support",
    ],
)

cc_library(
    name = "tf_saved_model_passes",
    srcs = [
        "convert_session_initializer_to_function.cc",
        "deduplicate_bound_input_bindings.cc",
        "freeze_global_tensors.cc",
        "freeze_saved_model_assets.cc",
        "lower_globals_to_ml_program.cc",
        "lower_variable_ops_to_ml_program.cc",
        "optimize_global_tensors.cc",
        "remove_vars_in_session_initializer.cc",
        "strip_saved_module_metadata.cc",
        "unfreeze_mutable_global_tensors.cc",
    ],
    hdrs = [
        "tf_saved_model_passes.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":machina_passes",
        ":tf_saved_model_asset_sinking_pass",
        ":tf_saved_model_freeze_utils",
        "//machina/compiler/mlir/machina",
        "//machina/compiler/mlir/machina:resource_value_typed_analyzer",
        "//machina/compiler/mlir/machina:machina_analysis",
        "//machina/compiler/mlir/machina:machina_ops",
        "//machina/compiler/mlir/machina:machina_types",
        "//machina/core:core_cpu",
        "//machina/core:framework_internal",
        "//machina/core:lib",
        "@com_google_absl//absl/strings",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:AffineUtils",
        "@toolchain-project//mlir:Analysis",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:MLProgramDialect",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:Support",
        "@toolchain-project//mlir:TransformUtils",
        "@toolchain-project//mlir:Transforms",
    ],
)

gentbl_cc_library(
    name = "tf_pass_inc_gen",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = {
        "tf_passes.h.inc": [
            "-gen-pass-decls",
            "-name=TensorFlow",
        ],
        "g3doc/_includes/tf_passes.md": ["-gen-pass-doc"],
    },
    tblgen = "@toolchain-project//mlir:mlir-tblgen",
    td_file = "tf_passes.td",
    deps = [
        "@toolchain-project//mlir:PassBaseTdFiles",
    ],
)

gentbl_cc_library(
    name = "tf_device_pass_inc_gen",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = {
        "tf_device_passes.h.inc": [
            "-gen-pass-decls",
            "-name=TensorFlowDevice",
        ],
        "g3doc/includes/tf_device_passes.md": ["-gen-pass-doc"],
    },
    tblgen = "@toolchain-project//mlir:mlir-tblgen",
    td_file = "tf_device_passes.td",
    deps = [
        "@toolchain-project//mlir:PassBaseTdFiles",
    ],
)

gentbl_cc_library(
    name = "tf_savedmodel_pass_inc_gen",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = {
        "tf_savedmodel_passes.h.inc": [
            "-gen-pass-decls",
            "-name=TensorFlowSavedModel",
        ],
        "g3doc/includes/tf_savedmodel_passes.md": ["-gen-pass-doc"],
    },
    tblgen = "@toolchain-project//mlir:mlir-tblgen",
    td_file = "tf_savedmodel_passes.td",
    deps = [
        "@toolchain-project//mlir:PassBaseTdFiles",
    ],
)

gentbl_cc_library(
    name = "tf_test_passes_inc_gen",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = {
        "test_passes.h.inc": [
            "-gen-pass-decls",
            "-name=TensorFlowTest",
        ],
        "g3doc/includes/tf_test_passes.md": ["-gen-pass-doc"],
    },
    tblgen = "@toolchain-project//mlir:mlir-tblgen",
    td_file = "tf_test_passes.td",
    deps = [
        "@toolchain-project//mlir:PassBaseTdFiles",
    ],
)

cc_library(
    name = "machina_passes",
    srcs = [
        "add_functions_for_exported_names.cc",
        "annotate_parameter_replication.cc",
        "batchmatmul_to_einsum.cc",
        "breakup-islands.cc",
        "canonicalize_compile_and_replicate_attributes.cc",
        "check_control_dependencies.cc",
        "cluster_ops_by_policy.cc",
        "cluster_outlining.cc",
        "cluster_tf_ops_pass.cc",
        "collection_ops_util.cc",
        "colocate_tpu_copy_with_dynamic_shape.cc",
        "constant_op_device_assignment.cc",
        "convert_control_to_data_outputs.cc",
        "convert_launch_func_to_tf_call.cc",
        "convert_tf_control_flow_to_scf.cc",
        "convert_to_legacy_compile_and_replicate_attributes.cc",
        "decompose_optionals.cc",
        "decompose_reduce_dataset.cc",
        "decompose_resource_ops_pass.cc",
        "device_attribute_to_launch.cc",
        "device_index_selector.cc",
        "drop_while_shape_invariant.cc",
        "einsum.cc",
        "executor_island_coarsening.cc",
        "executor_tpuv1_inline_tpu_island.cc",
        "executor_tpuv1_island_coarsening.cc",
        "executor_tpuv1_outline_tpu_island.cc",
        "extract_tpu_copy_with_dynamic_shape_op.cc",
        "fold_broadcast.cc",
        "functional_control_flow_to_cfg.cc",
        "functional_control_flow_to_regions.cc",
        "fused_kernel_matcher.cc",
        "generated_canonicalize.inc",
        "generated_optimize.inc",
        "gpu_fusion.cc",
        "graph_pruning.cc",
        "group_by_dialect.cc",
        "guarantee_all_funcs_one_use.cc",
        "hoist_loop_invariant.cc",
        "hoist_replicate_invariant_resource_writes.cc",
        "init_text_file_to_import.cc",
        "launch_to_device_attribute.cc",
        "layout_optimization.cc",
        "localize_var_handles.cc",
        "lower_quantized.cc",
        "mark_input_output_aliases.cc",
        "materialize_mlir_passthrough_op.cc",
        "merge_control_flow.cc",
        "move_tpu_compile_to_front.cc",
        "name_anonymous_iterators.cc",
        "optimize.cc",
        "order_by_dialect.cc",
        "parallel_execute_to_islands.cc",
        "prepare_tpu_computation_for_tf_export.cc",
        "print.cc",
        "promote_resources_to_args.cc",
        "readonly_references_to_resources.cc",
        "region_control_flow_to_functional.cc",
        "remove_unused_arguments.cc",
        "remove_unused_while_results.cc",
        "replica_id_to_device_ordinal.cc",
        "replicate_invariant_op_hoisting.cc",
        "replicate_tensor_list_init_ops_pass.cc",
        "replicate_to_island.cc",
        "resource_device_inference.cc",
        "resource_op_lifting.cc",
        "resource_op_lifting_cleanup.cc",
        "resource_op_lifting_cleanup.h",
        "rewrite_tpu_embedding_ops.cc",
        "sink_constant.cc",
        "stack_ops_decomposition.cc",
        "strip_noinline_attribute.cc",
        "strip_tf_attributes.cc",
        "tensor_array_ops_decomposition.cc",
        "tensor_device_copy_conversion.cc",
        "tensor_list_ops_decomposition.cc",
        "test_resource_alias_analysis.cc",
        "tf_data_optimization_pass.cc",
        "tf_device_assignment.cc",
        "tf_executor_to_functional.cc",
        "tf_functional_to_executor.cc",
        "tpu_annotate_dynamic_shape_inputs.cc",
        "tpu_cluster_cleanup_attributes.cc",
        "tpu_colocate_composite_resource_ops.cc",
        "tpu_colocate_splits.cc",
        "tpu_device_propagation.cc",
        "tpu_dynamic_layout_pass.cc",
        "tpu_host_computation_expansion.cc",
        "tpu_identity_pruning.cc",
        "tpu_parallel_execute_sink_resource_write.cc",
        "tpu_partitioned_op_conversion.cc",
        "tpu_reorder_replicate_and_partitioned_inputs.cc",
        "tpu_resource_partitioning.cc",
        "tpu_resource_read_for_write.cc",
        "tpu_space_to_depth_pass.cc",
        "tpu_update_embedding_enqueue_op_inputs.cc",
        "update_control_dependencies.cc",
        "verify_suitable_for_graph_export_pass.cc",
        "xla_call_module_deserialization.cc",
        "xla_call_module_serialization.cc",
        "xla_inline_device_ops.cc",
        "xla_rewrite.cc",
        "xla_validate_inputs.cc",
    ],
    hdrs = [
        "cluster_ops_by_policy.h",
        "collection_ops_util.h",
        "einsum.h",
        "passes.h",
    ],
    includes = ["include"],
    textual_hdrs = [
        "tf_device_passes.h.inc",
        "tf_passes.h.inc",
        "tf_savedmodel_passes.h.inc",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":cluster_formation",
        ":decompose_resource_ops",
        ":decompose_resource_ops_inc_gen",
        ":lower_tf_lib",
        ":shape_inference_pass",
        ":machina_optimize_inc_gen",
        ":tf_data_optimization",
        ":tf_device_pass_inc_gen",
        ":tf_pass_inc_gen",
        ":tf_savedmodel_pass_inc_gen",
        ":tfe_legalize_tfg",
        ":unroll_batch_matmul_pass",
        ":verify_no_outside_compilation_markers_pass",
        "//machina/compiler/jit:flags_headers",
        "//machina/compiler/mlir:op_or_arg_name_mapper",
        "//machina/compiler/mlir/machina",
        "//machina/compiler/mlir/machina:attribute_utils",
        "//machina/compiler/mlir/machina:bridge_logger",
        "//machina/compiler/mlir/machina:call_graph_util",
        "//machina/compiler/mlir/machina:cluster_util",
        "//machina/compiler/mlir/machina:convert_tensor",
        "//machina/compiler/mlir/machina:convert_type",
        "//machina/compiler/mlir/machina:device_util",
        "//machina/compiler/mlir/machina:dump_mlir_util",
        "//machina/compiler/mlir/machina:dynamic_shape_utils",
        "//machina/compiler/mlir/machina:error_util",
        "//machina/compiler/mlir/machina:mangling_util",
        "//machina/compiler/mlir/machina:parallel_execute_util",
        "//machina/compiler/mlir/machina:serialize_mlir_module_utils",
        "//machina/compiler/mlir/machina:stablehlo_custom_call_utils",
        "//machina/compiler/mlir/machina:string_util",
        "//machina/compiler/mlir/machina:machina_analysis",
        "//machina/compiler/mlir/machina:machina_op_interfaces",
        "//machina/compiler/mlir/machina:machina_ops",
        "//machina/compiler/mlir/machina:machina_side_effects",
        "//machina/compiler/mlir/machina:machina_traits",
        "//machina/compiler/mlir/machina:machina_types",
        "//machina/compiler/mlir/machina:tf_ops_layout_helper",
        "//machina/compiler/mlir/machina:tf_saved_model_inc_gen",
        "//machina/compiler/mlir/machina:topological_sort",
        "//machina/compiler/mlir/machina:tpu_cluster_util",
        "//machina/compiler/mlir/machina:tpu_embedding_ops_registry",
        "//machina/compiler/mlir/machina:tpu_rewrite_device_util",
        "//machina/compiler/mlir/machina:translate_utils",
        "//machina/compiler/mlir/machina:verification_utils",
        "//machina/compiler/mlir/machina:verify_suitable_for_graph_export",
        "//machina/compiler/mlir/machina:visitor",
        "//machina/compiler/mlir/machina:xla_call_module_attrs",
        "//machina/compiler/mlir/machina:xla_rewrite_util",
        "//machina/compiler/mlir/machina:xla_sharding_util",
        "//machina/compiler/mlir/machina/transforms/host_runtime:tpu_metadata_utils",
        "//machina/compiler/mlir/tf2xla/internal/inference:inference_metrics_pass",
        "//machina/compiler/mlir/tf2xla/transforms:legalization_op_config",
        "//machina/compiler/mlir/tf2xla/transforms:split_into_island_per_op_pass",
        "//machina/compiler/mlir/tf2xla/transforms:xla_legalize_tf",
        "//machina/compiler/mlir/tf2xla/transforms:xla_legalize_tf_with_tf2xla",
        "//machina/compiler/mlir/utils:validators",
        "//machina/compiler/tf2xla:side_effect_util",
        "//machina/compiler/tf2xla/kernels:xla_call_module_loader",
        "//machina/core:core_cpu_base",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core/ir/types:Dialect",
        "//machina/core/platform:error_payloads",
        "//machina/core/platform:logging",
        "//machina/core/platform:random",
        "//machina/core/protobuf/tpu:compile_metadata_proto_cc",
        "//machina/core/protobuf/tpu:dynamic_padding_proto_cc",
        "//machina/core/tpu:tpu_embedding_optimization_parameters_utils",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:variant",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:AffineAnalysis",
        "@toolchain-project//mlir:AffineUtils",
        "@toolchain-project//mlir:Analysis",
        "@toolchain-project//mlir:ArithDialect",
        "@toolchain-project//mlir:ControlFlowDialect",
        "@toolchain-project//mlir:ControlFlowInterfaces",
        "@toolchain-project//mlir:Dialect",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:FuncExtensions",
        "@toolchain-project//mlir:FunctionInterfaces",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:InferTypeOpInterface",
        "@toolchain-project//mlir:InliningUtils",
        "@toolchain-project//mlir:Parser",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:QuantOps",
        "@toolchain-project//mlir:Rewrite",
        "@toolchain-project//mlir:SCFDialect",
        "@toolchain-project//mlir:SideEffectInterfaces",
        "@toolchain-project//mlir:Support",
        "@toolchain-project//mlir:TensorDialect",
        "@toolchain-project//mlir:TransformUtils",
        "@toolchain-project//mlir:Transforms",
        "@local_tsl//tsl/platform:error_logging",
        "@local_xla//xla:window_util",
        "@local_xla//xla:xla_data_proto_cc",
        "@local_xla//xla:xla_proto_cc",
        "@local_xla//xla/hlo/builder:sharding_builder",
        "@local_xla//xla/mlir_hlo",
        "@shardy//shardy/dialect/sdy/ir:dialect",
        "@stablehlo//:chlo_ops",
        "@stablehlo//:stablehlo_ops",
        "@stablehlo//:stablehlo_portable_api",
        "@stablehlo//:stablehlo_serialization",
        "@stablehlo//:version",
        "@stablehlo//:vhlo_ops",
    ],
)

cc_library(
    name = "cluster_formation",
    srcs = ["cluster_formation.cc"],
    textual_hdrs = [
        "tf_passes.h.inc",
    ],
    deps = [
        ":tf_pass_inc_gen",
        "//machina/compiler/mlir/machina",
        "//machina/compiler/mlir/machina:attribute_utils",
        "//machina/compiler/mlir/machina:cluster_util",
        "//machina/compiler/mlir/machina:string_util",
        "//machina/compiler/mlir/machina:machina_analysis",
        "//machina/compiler/mlir/machina:tpu_rewrite_device_util",
        "//machina/core:framework",
        "//machina/core:portable_gif_internal",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:Support",
        "@toolchain-project//mlir:TransformUtils",
    ],
)

cc_library(
    name = "bridge",
    srcs = ["bridge.cc"],
    hdrs = ["bridge.h"],
    deps = [
        ":machina_passes",
        "//machina/compiler/mlir/machina:dump_mlir_util",
        "//machina/compiler/mlir/machina:error_util",
        "//machina/compiler/mlir/tf2xla/internal:logging_hooks",
        "//machina/core:lib_proto_parsing",
        "//machina/core/platform:status",
        "//machina/core/protobuf:for_core_protos_cc",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Pass",
    ],
)

cc_library(
    name = "verify_no_outside_compilation_markers_pass",
    srcs = ["verify_no_outside_compilation_markers_pass.cc"],
    hdrs = ["passes.h"],
    textual_hdrs = [
        "tf_device_passes.h.inc",
        "tf_passes.h.inc",
    ],
    deps = [
        ":tf_device_pass_inc_gen",
        ":tf_pass_inc_gen",
        "//machina/compiler/mlir/machina",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:Dialect",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:FuncExtensions",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:Support",
        "@shardy//shardy/dialect/sdy/ir:dialect",
    ],
)

tf_cc_test(
    name = "verify_no_outside_compilation_markers_pass_test",
    srcs = ["verify_no_outside_compilation_markers_pass_test.cc"],
    deps = [
        ":machina_passes",
        "//machina/compiler/mlir/tf2xla/transforms:test_utils",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:Support",
        "@local_xla//xla/tsl/platform:statusor",
    ],
)

cc_library(
    name = "shape_inference_pass",
    srcs = [
        "passes.h",
        "shape_inference.cc",
        "shape_inference_pass.cc",
    ],
    hdrs = [
        "shape_inference.h",
    ],
    deps = [
        ":tf_device_pass_inc_gen",
        ":tf_pass_inc_gen",
        "//machina/compiler/mlir/machina",
        "//machina/compiler/mlir/machina:dynamic_shape_utils",
        "//machina/compiler/mlir/machina:mlir_roundtrip_flags",
        "//machina/compiler/mlir/machina:serialize_mlir_module_utils",
        "//machina/compiler/mlir/machina:shape_inference_utils",
        "//machina/compiler/mlir/machina:translate_utils",
        "//machina/compiler/mlir/machina/translate/tools:parsers",
        "//machina/compiler/tf2xla/kernels:xla_call_module_loader",
        "//machina/core:framework",
        "//machina/core:protos_all_cc",
        "//machina/core/ir/types:Dialect",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:FuncExtensions",
        "@toolchain-project//mlir:FunctionInterfaces",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:InferTypeOpInterface",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:Support",
        "@toolchain-project//mlir:TensorDialect",
        "@toolchain-project//mlir:TransformUtils",
        "@local_xla//xla:shape_util",
        "@local_xla//xla:window_util",
        "@local_xla//xla:xla_data_proto_cc",
        "@local_xla//xla/hlo/translate/hlo_to_mhlo:hlo_utils",
        "@local_xla//xla/hlo/translate/mhlo_to_hlo:type_to_shape",
        "@local_xla//xla/mlir_hlo",
        "@local_xla//xla/service:shape_inference",
        "@local_xla//xla/tsl/platform:errors",
        "@local_xla//xla/tsl/util:env_var",
        "@shardy//shardy/dialect/sdy/ir:dialect",
        "@stablehlo//:stablehlo_serialization",
        "@stablehlo//:version",
    ],
)

cc_library(
    name = "machina_test_passes",
    testonly = True,  # Ensure alwayslink does not leak in the codebase.
    srcs = [
        "init_text_file_to_import_test_pass.cc",
        "initialize_variables_in_session_init_test_pass.cc",
        "lift_variables_test_pass.cc",
        "lower_tf_test_pass.cc",
        "mark_initialized_variables_test_pass.cc",
        "resource_analyzer_test_pass.cc",
        "test_cluster_ops_by_policy.cc",
        "test_passes.h.inc",
        "test_side_effect_analysis.cc",
        "tf_saved_model_freeze_variables_test_pass.cc",
    ],
    hdrs = [
        "test_passes.h",
    ],
    deps = [
        ":initialize_variables_in_session_init",
        ":lift_variables_lib",
        ":lower_tf_lib",
        ":mark_initialized_variables_lib",
        ":machina_passes",
        ":tf_saved_model_freeze_variables",
        ":tf_saved_model_passes",
        ":tf_test_passes_inc_gen",
        "//machina/compiler/mlir/machina",
        "//machina/compiler/mlir/machina:error_util",
        "//machina/compiler/mlir/machina:fake_session",
        "//machina/compiler/mlir/machina:resource_value_typed_analyzer",
        "//machina/compiler/mlir/machina:machina_analysis",
        "//machina/core:core_cpu",
        "//machina/core:core_cpu_lib",
        "//machina/core:framework",
        "//machina/core:framework_internal",
        "//machina/core:lib",
        "//machina/core/platform:errors",
        "//machina/core/platform:status",
        "//machina/core/platform:threadpool_options",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:ArithDialect",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:Support",
        "@toolchain-project//mlir:TransformUtils",
        "@toolchain-project//mlir:Transforms",
    ],
    alwayslink = 1,
)

cc_library(
    name = "graph_optimization_pass",
    srcs = ["graph_optimization_pass.cc"],
    hdrs = ["graph_optimization_pass.h"],
    deps = [
        ":machina_passes",
        "//machina/compiler/mlir:mlir_graph_optimization_pass",
        "//machina/compiler/mlir/machina:dump_mlir_util",
        "//machina/compiler/mlir/machina:error_util",
        "//machina/core/protobuf:for_core_protos_cc",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:Support",
        "@toolchain-project//mlir:Transforms",
    ],
)

cc_library(
    name = "graph_optimization_pass_registration",
    srcs = ["graph_optimization_pass_registration.cc"],
    deps = [
        ":graph_optimization_pass",
        "//machina/compiler/mlir:mlir_graph_optimization_pass",
        "//machina/compiler/mlir:mlir_graph_optimization_pass_registration",
    ],
)

cc_library(
    name = "constant_fold_utils",
    srcs = [
        "constant_fold_utils.cc",
    ],
    hdrs = [
        "constant_fold_utils.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//machina/compiler/mlir/machina",
        "//machina/compiler/mlir/machina:convert_tensor",
        "//machina/compiler/mlir/machina:machina_traits",
        "//machina/compiler/mlir/machina:translate_utils",
        "//machina/core:protos_all_cc",
        "//machina/core/tfrt/fallback:fallback_state",
        "//machina/core/tfrt/fallback:op_kernel_runner",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Support",
    ],
)

cc_library(
    name = "tf_dialect_passes",
    srcs = [
        "constant_fold.cc",
        "decode_attributes_hook.cc",
    ],
    hdrs = [
        "constant_fold.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":constant_fold_utils",
        "//machina/compiler/mlir/machina",
        "//machina/compiler/mlir/machina:convert_tensor",
        "//machina/compiler/mlir/machina:machina_types",
        "//machina/core:all_kernels",
        "//machina/core:direct_session",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core/ops",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Support",
    ],
    alwayslink = 1,
)

cc_library(
    name = "tf_dialect_lib",
    deps = [
        ":tf_dialect_passes",
        "@toolchain-project//mlir:AllPassesAndDialects",
    ],
)

cc_library(
    name = "tf_graph_optimization_pass",
    srcs = ["tf_graph_optimization_pass.cc"],
    hdrs = ["tf_graph_optimization_pass.h"],
    deps = [
        "//machina/compiler/mlir/machina",
        "//machina/compiler/mlir/tf2xla/api/v2:graph_to_tf_executor",
        "//machina/compiler/mlir/tf2xla/api/v2:tf_executor_to_graph",
        "//machina/compiler/tf2xla:functionalize_control_flow_pass_registration",
        "//machina/core:core_cpu",
        "//machina/core:framework",
        "//machina/core:ops",
        "//machina/core:protos_all_cc",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Pass",
    ],
)

filegroup(
    name = "machina_optimize_td_files",
    srcs = [
        "optimize.td",
    ],
)

gentbl_cc_library(
    name = "machina_optimize_inc_gen",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = {"generated_optimize.inc": ["-gen-rewriters"]},
    tblgen = "@toolchain-project//mlir:mlir-tblgen",
    td_file = "optimize.td",
    deps = [
        "//machina/compiler/mlir/machina:machina_ops_td_files",
        "@toolchain-project//mlir:ArithOpsTdFiles",
        "@toolchain-project//mlir:FuncTdFiles",
    ],
)

# This is a separate library so that external passes can link only this library
# without linking any of the other machina passes.
gentbl_cc_library(
    name = "lower_tf_inc_gen",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = {"generated_lower_tf.inc": ["-gen-rewriters"]},
    tblgen = "@toolchain-project//mlir:mlir-tblgen",
    td_file = "lower_tf.td",
    deps = [
        ":rewrite_util_td_files",
        "//machina/compiler/mlir/machina:machina_ops_td_files",
        "@toolchain-project//mlir:FuncTdFiles",
    ],
)

cc_library(
    name = "lower_tf_lib",
    srcs = [
        "lower_tf.cc",
    ],
    hdrs = [
        "lower_tf.h",
    ],
    deps = [
        ":lower_tf_inc_gen",
        ":rewrite_util",
        "//machina/compiler/mlir/machina",
        "//machina/compiler/mlir/machina:dynamic_shape_utils",
        "//machina/compiler/mlir/machina:machina_ops",
        "//machina/compiler/mlir/machina:machina_types",
        "//machina/core:framework",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Support",
    ],
)

cc_library(
    name = "set_tpu_infeed_layout",
    srcs = ["set_tpu_infeed_layout.cc"],
    hdrs = ["set_tpu_infeed_layout.h"],
    deps = [
        "//machina/compiler/mlir/machina",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Support",
        "@local_xla//xla:shape_util",
        "@local_xla//xla/hlo/translate/mhlo_to_hlo:type_to_shape",
        "@local_xla//xla/mlir_hlo",
        "@local_xla//xla/stream_executor/tpu:c_api_conversions",
        "@local_xla//xla/stream_executor/tpu:tpu_api",
    ],
)

cc_library(
    name = "tf_saved_model_asset_sinking_pass",
    srcs = ["tf_saved_model_asset_sinking_pass.cc"],
    hdrs = ["tf_saved_model_asset_sinking_pass.h"],
    deps = [
        ":tf_savedmodel_pass_inc_gen",
        "//machina/compiler/mlir/machina",
        "//machina/compiler/mlir/machina:machina_types",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:Support",
        "@local_tsl//tsl/platform:path",
    ],
)

tf_cc_test(
    name = "shape_inference_test",
    srcs = ["shape_inference_test.cc"],
    deps = [
        ":shape_inference_pass",
        "@com_google_googletest//:gtest_main",
        "@toolchain-project//mlir:IR",
        "@local_xla//xla/tsl/platform:env",
    ],
)

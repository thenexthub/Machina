load("@toolchain-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//machina:machina.bzl", "if_google", "tf_cc_binary", "tf_cc_test")

# Note: keep the following lines separate due to the way copybara works
load("//machina:machina.default.bzl", "get_compatible_with_portable")
load(
    "//machina/core/platform:build_config.bzl",
    "tf_proto_library",
)

# TF to TFRT kernels conversion.
package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = [":friends"],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    packages = [
        "//machina/compiler/...",
        "//machina/core/tfrt/experimental/data/...",
        "//machina/core/tfrt/graph_executor/...",
        "//machina/core/tfrt/ifrt/...",
        "//machina/core/tfrt/kernels/...",
        "//machina/core/tfrt/mlrt/...",
        "//machina/core/tfrt/saved_model/...",
        "//machina/core/tfrt/tfrt_session/...",
    ] + if_google([
        "//learning/brain/experimental/mlir/tflite/tfmrt/...",
        "//learning/brain/experimental/mlir/tfrt_compiler/...",
        "//learning/brain/experimental/tfrt/...",
        "//learning/brain/tfrt/...",
        "//learning/infra/mira/...",
        "//learning/serving/contrib/tfrt/mlir/...",
        "//learning/serving/testing/functional/...",
        # Allow visibility from the mlir language server.
        "//learning/brain/mlir/mlir_lsp_server/...",
        "//smartass/brain/ops/...",
        "//third_party/mlir_edge/tpgen/...",
        "//machina_serving/servables/machina/...",
        "//machina_serving/servables/machina/google/...",
        "//third_party/tf_runtime_google/...",
    ]),
)

exports_files(["run_lit.sh"])

td_library(
    name = "runtime_fallback_ops_td_files",
    srcs = [
        "runtime_fallback/runtime_fallback_ops.td",
    ],
    deps = [
        "@toolchain-project//mlir:OpBaseTdFiles",
        "@toolchain-project//mlir:SideEffectInterfacesTdFiles",
        "@tf_runtime//:OpBaseTdFiles",
    ],
)

gentbl_cc_library(
    name = "runtime_fallback_ops_inc_gen",
    tbl_outs = {
        "runtime_fallback_ops.h.inc": ["-gen-op-decls"],
        "runtime_fallback_ops.cc.inc": ["-gen-op-defs"],
    },
    tblgen = "@toolchain-project//mlir:mlir-tblgen",
    td_file = "runtime_fallback/runtime_fallback_ops.td",
    deps = [":runtime_fallback_ops_td_files"],
)

cc_library(
    name = "runtime_fallback_opdefs",
    srcs = [
        "runtime_fallback/runtime_fallback_combine.cc",
        "runtime_fallback/runtime_fallback_ops.cc",
    ],
    hdrs = [
        "runtime_fallback/runtime_fallback_ops.h",
    ],
    deps = [
        ":runtime_fallback_ops_inc_gen",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:SideEffectInterfaces",
        "@tf_runtime//:basic_kernels_opdefs",
        "@tf_runtime//:tensor_opdefs",
    ],
)

cc_library(
    name = "runtime_fallback_executor",
    testonly = True,
    srcs = [
        "runtime_fallback/runtime_fallback_executor.cc",
    ],
    hdrs = [
        "runtime_fallback/runtime_fallback_executor.h",
    ],
    deps = [
        ":host_context_util",
        ":tf_to_tfrt",
        "//machina/compiler/mlir/machina",
        "//machina/core:framework",
        "//machina/core/platform:env",
        "//machina/core/platform:threadpool_interface",
        "//machina/core/runtime_fallback/kernel:kernel_fallback_execute_compat_eager",
        "//machina/core/runtime_fallback/runtime:kernel_utils",
        "//machina/core/runtime_fallback/runtime:runtime_fallback_alwayslink",
        "//machina/core/tfrt/utils:fallback_tensor",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:Parser",
        "@toolchain-project//mlir:Pass",
        "@tf_runtime//:basic_kernels_alwayslink",
        "@tf_runtime//:bef",
        "@tf_runtime//:befexecutor",
        "@tf_runtime//:core_runtime_alwayslink",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:mlirtobef",
        "@tf_runtime//:support",
    ],
)

cc_library(
    name = "corert_converter",
    srcs = [
        "transforms/corert_converter.cc",
    ],
    hdrs = [
        "transforms/corert_converter.h",
    ],
    compatible_with = get_compatible_with_portable(),  # copybara: comment
    deps = [
        ":attr_lowering_utils",
        ":transform_utils",
        "//machina/compiler/mlir/machina:machina_analysis",
        "//machina/compiler/mlir/machina:machina_types",
        "//machina/core:framework",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:DerivedAttributeOpInterface",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:Support",
        "@toolchain-project//mlir:TransformUtils",
        "@tf_runtime//:basic_kernels_opdefs",
        "@tf_runtime//:core_runtime_opdefs",
    ],
)

cc_library(
    name = "fallback_converter",
    srcs = [
        "transforms/fallback_converter.cc",
    ],
    hdrs = [
        "transforms/fallback_converter.h",
    ],
    compatible_with = get_compatible_with_portable(),  # copybara: comment
    deps = [
        "//machina/compiler/mlir/machina:machina_types",
        "//machina/compiler/mlir/tfrt/ir:tfrt_fallback_async_opdefs",
        "//machina/compiler/mlir/tfrt/ir:tfrt_fallback_opdefs",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Support",
        "@toolchain-project//mlir:TransformUtils",
        "@tf_runtime//:basic_kernels_opdefs",
        "@tf_runtime//:core_runtime_opdefs",
    ],
)

cc_library(
    name = "transforms/gpu_passes",
    srcs = ["transforms/gpu_passes.cc"],
    hdrs = ["transforms/gpu_passes.h"],
    deps = [
        "//machina/compiler/mlir/tfrt/ir:tfrt_gpu_opdefs",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:TransformUtils",
    ],
)

cc_library(
    name = "tfrt_pipeline_options",
    hdrs = ["transforms/tfrt_pipeline_options.h"],
    compatible_with = get_compatible_with_portable(),  # copybara: comment
    deps = [
        ":tfrt_compile_options",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:Pass",
    ],
)

cc_library(
    name = "tf_to_tfrt",
    srcs = [
        "transforms/cross_device_transfer.cc",
        "transforms/deduplicate_batch_function.cc",
        "transforms/deduplicate_if_result_pass.cc",
        "transforms/fuse_tpu_compile_and_execute_ops.cc",
        "transforms/insert_tensor_copy.cc",
        "transforms/lower_saved_model.cc",
        "transforms/merge_tf_if_ops.cc",
        "transforms/optimize.cc",
        "transforms/optimize_tf_control_flow_side_effect.cc",
        "transforms/passes.cc",
        "transforms/reconfig_batch_op.cc",
        "transforms/remove_device_attribute.cc",
        "transforms/remove_tf_if_const_args.cc",
        "transforms/reorder_assert.cc",
        "transforms/sink_in_invariant_ops.cc",
        "transforms/tf_to_tfrt.cc",
        "transforms/xla_rewrite_pass.cc",
    ],
    hdrs = [
        "transforms/passes.h",
    ],
    deps = [
        ":attr_lowering_utils",
        ":corert_converter",
        ":cost_analysis",
        ":fallback_converter",
        ":tensor_array_side_effect_analysis",
        ":tfrt_compile_options",
        ":tfrt_pipeline_options",
        ":tpu_passes",
        ":transform_utils",
        ":transforms/gpu_passes",
        ":transforms/set_shape_invariant_in_while_ops",
        "//machina/compiler/mlir/machina",
        "//machina/compiler/mlir/machina:bridge_logger",
        "//machina/compiler/mlir/machina:convert_tensor",
        "//machina/compiler/mlir/machina:machina_analysis",
        "//machina/compiler/mlir/machina:machina_op_interfaces",
        "//machina/compiler/mlir/machina:machina_ops",
        "//machina/compiler/mlir/machina:machina_types",
        "//machina/compiler/mlir/machina/ir/host_runtime:machina_tfrt_ops",
        "//machina/compiler/mlir/machina/transforms:machina_passes",
        "//machina/compiler/mlir/machina/transforms:tf_saved_model_asset_sinking_pass",
        "//machina/compiler/mlir/tfrt/ir:tfrt_fallback_async_opdefs",
        "//machina/compiler/mlir/tfrt/ir:tfrt_fallback_opdefs",
        "//machina/compiler/mlir/tfrt/ir:tfrt_gpu_opdefs",
        "//machina/compiler/tf2xla:tf2xla_defs",
        "//machina/core:framework",
        "//machina/core/ir/types:Dialect",
        "//machina/core/platform:errors",
        "//machina/core/platform:status",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:FuncExtensions",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:Rewrite",
        "@toolchain-project//mlir:SideEffectInterfaces",
        "@toolchain-project//mlir:Support",
        "@toolchain-project//mlir:TransformUtils",
        "@toolchain-project//mlir:Transforms",
        "@local_tsl//tsl/platform:errors",
        "@tf_runtime//:basic_kernels_opdefs",
        "@tf_runtime//:core_runtime_opdefs",
        "@tf_runtime//:stream_analysis",
        "@tf_runtime//:test_kernels_opdefs",
    ],
    alwayslink = 1,
)

cc_library(
    name = "host_context_util",
    srcs = ["utils/host_context.cc"],
    hdrs = ["utils/host_context.h"],
    deps = [
        "//machina/core/platform:logging",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log",
        "@tf_runtime//:hostcontext",
    ],
)

cc_library(
    name = "function",
    srcs = [
        "function/function.cc",
    ],
    hdrs = [
        "function/function.h",
    ],
    deps = [
        ":tf_to_tfrt",
        ":tfrt_compile_options",
        ":tfrt_pipeline_options",
        "//machina/compiler/mlir/machina:dump_mlir_util",
        "//machina/compiler/mlir/machina:error_util",
        "//machina/compiler/mlir/machina:translate_lib",
        "//machina/core/platform:errors",
        "//machina/core/platform:status",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:Support",
        "@tf_runtime//:bef",
        "@tf_runtime//:core_runtime",
        "@tf_runtime//:mlirtobef",
    ],
)

cc_library(
    name = "saved_model",
    srcs = [
        "saved_model/saved_model.cc",
    ],
    hdrs = [
        "saved_model/saved_model.h",
    ],
    deps = [
        ":tf_to_tfrt",
        "//machina/compiler/mlir/machina",
        "//machina/compiler/mlir/machina:convert_type",
        "//machina/compiler/mlir/machina:translate_lib",
        "//machina/compiler/mlir/machina/transforms:tf_dialect_passes",
        "//machina/core:framework",
        "//machina/core:protos_all_cc",
        "//machina/core/platform:status",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:Support",
        "@local_tsl//tsl/platform:errors",
        "@tf_runtime//:bef",
        "@tf_runtime//:core_runtime",
        "@tf_runtime//:mlirtobef",
    ],
)

cc_library(
    name = "import_model",
    srcs = [
        "translate/import_model.cc",
    ],
    hdrs = [
        "translate/import_model.h",
    ],
    visibility = [
        "//machina/compiler/mlir/tfrt/tests/saved_model:__pkg__",
        "//machina/compiler/mlir/tfrt/transforms/mlrt:__pkg__",
        "//machina/core/tfrt/graph_executor:__pkg__",
        "//machina/core/tfrt/saved_model:__pkg__",
        "//machina/core/tfrt/saved_model/utils:__pkg__",
    ],
    deps = [
        ":backend_compiler",
        ":function",
        ":tf_to_tfrt",
        ":tfrt_compile_options",
        ":tfrt_pipeline_options",
        ":tpu_passes",
        "//machina/compiler/mlir/machina",
        "//machina/compiler/mlir/machina:dump_mlir_util",
        "//machina/compiler/mlir/machina:error_util",
        "//machina/compiler/mlir/machina:mlir_roundtrip_flags",
        "//machina/compiler/mlir/machina:serialize_mlir_module_utils",
        "//machina/compiler/mlir/machina/transforms:machina_passes",
        "//machina/compiler/mlir/machina/transforms:tf_saved_model_asset_sinking_pass",
        "//machina/compiler/mlir/machina/transforms/host_runtime:lower_cluster_to_runtime_ops",
        "//machina/compiler/mlir/tf2xla/api/v2:cluster_tf",
        "//machina/compiler/mlir/tf2xla/api/v2:tf_dialect_to_executor",
        "//machina/compiler/mlir/tf2xla/api/v2:tf_executor_to_graph",
        "//machina/compiler/tf2xla:xla_op_registry",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core/common_runtime:function_def_utils",
        "//machina/core/platform:status",
        "//machina/core/tfrt/fallback:fallback_state",
        "//machina/core/tfrt/runtime",
        "//machina/core/tpu:tpu_defs",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:FuncExtensions",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:Support",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:errors",
        "@tf_runtime//:bef",
        "@tf_runtime//:mlirtobef",
        "@tf_runtime//:support",
    ],
)

tf_cc_test(
    name = "import_model_test",
    srcs = [
        "translate/import_model_test.cc",
    ],
    tags = ["no_oss"],
    deps = [
        ":import_model",
        ":tfrt_compile_options",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "tfrt_compile_options",
    srcs = ["translate/tfrt_compile_options.cc"],
    hdrs = ["translate/tfrt_compile_options.h"],
    compatible_with = get_compatible_with_portable(),  # copybara: comment
    visibility = ["//visibility:public"],
    deps = [
        "//machina/core/protobuf:for_core_protos_cc",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "cost_analysis",
    srcs = ["analysis/cost_analysis.cc"],
    hdrs = ["analysis/cost_analysis.h"],
    deps = [
        ":constants",
        "//machina/compiler/mlir/machina",
        "//machina/compiler/mlir/machina:machina_types",
        "//machina/core/platform:status",
        "//machina/core/tfrt/fallback:cost_recorder",
        "//machina/core/tfrt/fallback:op_cost_map_proto_cc",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Support",
        "@tf_runtime//:compiler_tfrt_op_interfaces",
    ],
)

cc_library(
    name = "test_cost_analysis_pass",
    srcs = ["analysis/test_cost_analysis_pass.cc"],
    deps = [
        ":cost_analysis",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:Pass",
    ],
    alwayslink = 1,
)

cc_library(
    name = "tensor_array_side_effect_analysis",
    srcs = ["analysis/tensor_array_side_effect_analysis.cc"],
    hdrs = ["analysis/tensor_array_side_effect_analysis.h"],
    deps = [
        "//machina/compiler/mlir/machina",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:SideEffectInterfaces",
    ],
)

cc_library(
    name = "test_tensor_array_side_effect_analysis",
    srcs = ["analysis/test_tensor_array_side_effect_analysis.cc"],
    deps = [
        ":tensor_array_side_effect_analysis",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:Support",
    ],
    alwayslink = 1,
)

tf_proto_library(
    name = "analysis/analysis_proto",
    srcs = ["analysis/analysis.proto"],
)

cc_library(
    name = "passes",
    visibility = ["//visibility:private"],
    deps = [
        "//machina/compiler/mlir/tfrt:tf_to_tfrt",
    ],
)

cc_library(
    name = "transforms/set_shape_invariant_in_while_ops",
    srcs = ["transforms/set_shape_invariant_in_while_ops.cc"],
    hdrs = ["transforms/set_shape_invariant_in_while_ops.h"],
    deps = [
        "//machina/compiler/mlir/machina",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:Support",
        "@toolchain-project//mlir:Transforms",
    ],
)

cc_library(
    name = "tf_tfrt_opt_lib",
    testonly = True,
    srcs = ["tf-tfrt-opt.cc"],
    deps = [
        ":passes",
        ":test_cost_analysis_pass",
        ":test_opkernels",
        ":test_tensor_array_side_effect_analysis",
        ":tf_to_tfrt",
        ":tpu_passes",
        ":transforms/gpu_passes",
        "//machina/compiler/mlir:init_mlir",
        "//machina/compiler/mlir:passes",
        "//machina/compiler/mlir/machina",
        "//machina/compiler/mlir/machina/transforms:machina_passes",
        "//machina/compiler/mlir/machina/transforms/host_runtime:lower_cluster_to_runtime_ops",
        "//machina/compiler/mlir/tfrt/ir:tfrt_fallback_async_opdefs",
        "//machina/compiler/mlir/tfrt/ir:tfrt_fallback_opdefs",
        "//machina/compiler/mlir/tfrt/ir:tfrt_fallback_sync_opdefs",
        "//machina/compiler/mlir/tfrt/ir/mlrt:mlrt_ops",
        "//machina/compiler/mlir/tfrt/ir/mlrt:tf_mlrt_ops",
        "//machina/compiler/mlir/tfrt/transforms/ifrt:tf_ifrt_passes",
        "//machina/compiler/mlir/tfrt/transforms/mlrt:passes",
        "//machina/core:machina",
        "@toolchain-project//mlir:AllPassesAndDialects",
        "@toolchain-project//mlir:MlirOptLib",
        "@toolchain-project//mlir:RegisterAllDialects",
        "@toolchain-project//mlir:RegisterAllPasses",
        "@toolchain-project//mlir:ShapeDialect",
        "@toolchain-project//mlir:Support",
        "@toolchain-project//mlir:Transforms",
        "@local_xla//xla/mlir_hlo",
        "@stablehlo//:register",
        "@tf_runtime//:init_tfrt_dialects",
        "@tf_runtime//:print_stream_pass",
    ],
)

tf_cc_binary(
    name = "tf-tfrt-opt",
    testonly = True,
    deps = [":tf_tfrt_opt_lib"],
)

tf_cc_binary(
    name = "tfrt_translate",
    srcs = ["tools/tfrt_translate/static_registration.cc"],
    visibility = [":friends"],
    deps = [
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:TranslateLib",
        "@tf_runtime//:beftomlir_translate",
        "@tf_runtime//:init_tfrt_dialects",
        "@tf_runtime//:mlirtobef_translate",
    ] + if_google(
        [
            "//third_party/tf_runtime_llvm:tfrt_translate_main",
        ],
        ["@tf_runtime//third_party/llvm_derived:tfrt_translate_main"],
    ),
)

tf_cc_binary(
    name = "bef_executor",
    testonly = True,
    visibility = [":friends"],
    deps = [
        "@tf_runtime//:dtype",
        "@tf_runtime//:simple_tracing_sink",
        "@tf_runtime//tools:bef_executor_expensive_kernels",
        "@tf_runtime//tools:bef_executor_lib",
        "@tf_runtime//tools:bef_executor_lightweight_kernels",
    ],
)

cc_library(
    name = "tfrt_fallback_registration",
    srcs = [
        "tfrt_fallback_registration.cc",
    ],
    hdrs = [
        "tfrt_fallback_registration.h",
    ],
    visibility = [":friends"] + if_google([
        "//learning/brain/experimental/tfrt/visualization:__pkg__",
        # Allow visibility from the mlir language server.
        "//learning/brain/mlir/mlir_lsp_server:__pkg__",
    ]),
    deps = [
        "//machina/compiler/mlir/tfrt/ir:tfrt_fallback_async_opdefs",
        "//machina/compiler/mlir/tfrt/ir:tfrt_fallback_opdefs",
        "//machina/compiler/mlir/tfrt/ir:tfrt_fallback_sync_opdefs",
        "@toolchain-project//mlir:IR",
    ],
)

tf_cc_binary(
    name = "tfrt_fallback_translate",
    srcs = [
        "tfrt_fallback_translate_registration.cc",
    ],
    visibility = [":friends"],
    deps = [
        "//machina/compiler/mlir/machina",
        "//machina/compiler/mlir/tfrt:tfrt_fallback_registration",
        "//machina/compiler/mlir/tfrt/ir:tfrt_fallback_opdefs",
        "@toolchain-project//mlir:TranslateLib",
        "@tf_runtime//:init_tfrt_dialects",
        "@tf_runtime//:mlirtobef_translate",
    ] + if_google(
        [
            "//third_party/tf_runtime_llvm:tfrt_translate_main",
        ],
        ["@tf_runtime//third_party/llvm_derived:tfrt_translate_main"],
    ),
)

cc_library(
    name = "attr_lowering_utils",
    srcs = [
        "transforms/attr_lowering_utils.cc",
    ],
    hdrs = [
        "transforms/attr_lowering_utils.h",
    ],
    compatible_with = get_compatible_with_portable(),  # copybara: comment
    visibility = [":friends"],
    deps = [
        "//machina/compiler/mlir/machina:machina_attributes",
        "//machina/compiler/mlir/machina:machina_types",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Support",
        "@tf_runtime//:core_runtime_opdefs",
    ],
)

cc_library(
    name = "transform_utils",
    srcs = [
        "transforms/utils.cc",
    ],
    hdrs = [
        "transforms/utils.h",
    ],
    compatible_with = get_compatible_with_portable(),  # copybara: comment
    visibility = [":friends"],
    deps = [
        "//machina/compiler/mlir/machina",
        "//machina/compiler/mlir/machina:machina_ops",
        "//machina/compiler/mlir/machina/ir/host_runtime:machina_tfrt_ops",
        "@com_google_absl//absl/strings",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Support",
        "@tf_runtime//:basic_kernels_opdefs",
        "@tf_runtime//:core_runtime_opdefs",
    ],
)

cc_library(
    name = "transforms/update_op_cost_in_tfrt_mlir",
    srcs = ["transforms/update_op_cost_in_tfrt_mlir.cc"],
    hdrs = ["transforms/update_op_cost_in_tfrt_mlir.h"],
    deps = [
        ":cost_analysis",
        "//machina/core/tfrt/fallback:cost_recorder",
        "@toolchain-project//mlir:IR",
    ],
)

cc_library(
    name = "tpu_passes",
    hdrs = ["transforms/tpu_passes.h"],
    visibility = [
        ":friends",
    ] + if_google([
        "//learning/brain/tfrt/ifrt/pjrt/__subpackages__",
        "//learning/serving/servables/tfrt:__subpackages__",
        "//smartass/brain/util:__pkg__",
    ]),
    deps = [
        ":fallback_converter",
        ":tfrt_compile_options",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:Support",
        "@toolchain-project//mlir:TransformUtils",
    ],
)

cc_library(
    name = "constants",
    hdrs = ["constants.h"],
)

cc_library(
    name = "test_opkernels",
    testonly = True,
    srcs = ["test_opkernels.cc"],
    deps = ["//machina/core:framework"],
    alwayslink = True,
)

cc_library(
    name = "backend_compiler",
    srcs = ["backend_compiler.cc"],
    hdrs = ["backend_compiler.h"],
    deps = [
        "//machina/core/tfrt/runtime",
        "@com_google_absl//absl/status",
        "@toolchain-project//mlir:IR",
    ],
)

cc_library(
    name = "export",
    srcs = ["utils/export.cc"],
    hdrs = ["utils/export.h"],
    deps = [
        "//machina/compiler/mlir/machina:error_util",
        "//machina/compiler/mlir/machina:mlir_roundtrip_flags",
        "//machina/compiler/mlir/machina/transforms:machina_passes",
        "//machina/compiler/mlir/tf2xla/api/v1:tf_dialect_to_executor",
        "//machina/compiler/mlir/tf2xla/api/v2:tf_executor_to_graph",
        "//machina/core:protos_all_cc",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:Support",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/profiler/lib:traceme",
    ],
)

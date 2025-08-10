load("@toolchain-project//mlir:tblgen.bzl", "gentbl_cc_library")
load("//machina:machina.bzl", "tf_cc_test")
load("//machina:machina.default.bzl", "filegroup", "get_compatible_with_portable")
load("//machina/compiler/mlir:glob_lit_test.bzl", "glob_lit_tests")
load("//machina/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = [
        "//learning/brain/tfrt/tpu/compiler/mlir:__pkg__",
        "//machina/compiler/mlir:__subpackages__",
        "//machina/compiler/mlir/tf2xla/api:__subpackages__",
        "//machina/compiler/mlir/tf2xla/internal:__subpackages__",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "clustering_passes",
    hdrs = [
        "clustering_passes.h",
    ],
    textual_hdrs = [
        "clustering_passes.h.inc",
    ],
    deps = [
        ":extract_head_tail_outside_compilation",
        ":extract_outside_compilation",
        ":hoist_broadcast_read",
        ":mark_ops_for_outside_compilation",
        ":tpu_cluster_formation",
        ":tpu_sharding_identification_pass",
        ":tpu_validate_inputs",
        ":tpu_validate_session_inputs",
        ":verify_clustering_pass",
        ":xla_broadcast",
        ":xla_cluster_formation",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Pass",
    ],
)

cc_library(
    name = "verify_clustering_pass",
    srcs = [
        "verify_clustering_pass.cc",
    ],
    deps = [
        ":clustering_passes_inc_gen",
        "//machina/compiler/mlir/machina",
        "//machina/compiler/mlir/machina:attribute_utils",
        "//machina/compiler/mlir/machina:string_util",
        "//machina/compiler/mlir/machina:machina_analysis",
        "//machina/compiler/mlir/machina:tpu_rewrite_device_util",
        "//machina/compiler/mlir/tf2xla/internal/utils:dialect_detection_utils",
        "//machina/core:framework",
        "//machina/core/transforms/toposort:Pass",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:Support",
        "@toolchain-project//mlir:Transforms",
    ],
)

gentbl_cc_library(
    name = "clustering_passes_inc_gen",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = {"clustering_passes.h.inc": [
        "-gen-pass-decls",
        "-name=TFXLABridgeClustering",
    ]},
    tblgen = "@toolchain-project//mlir:mlir-tblgen",
    td_file = "clustering_passes.td",
    deps = [
        "@toolchain-project//mlir:PassBaseTdFiles",
    ],
)

tf_cc_test(
    name = "verify_clustering_pass_test",
    srcs = ["verify_clustering_pass_test.cc"],
    deps = [
        ":clustering_passes",
        "//machina/compiler/mlir/tf2xla/transforms:test_utils",
        "@com_google_googletest//:gtest_main",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:Support",
        "@local_xla//xla/tsl/platform:statusor",
    ],
)

glob_lit_tests(
    name = "all_tests",
    data = [":test_utilities"],
    driver = "@toolchain-project//mlir:run_lit.sh",
    test_file_exts = [
        "mlir",
    ],
)

# Bundle together all of the test utilities that are used by tests.
filegroup(
    name = "test_utilities",
    testonly = True,
    data = [
        "//machina/compiler/mlir:tf-opt",
        "@toolchain-project//toolchain:FileCheck",
    ],
)

cc_library(
    name = "tpu_cluster_formation",
    srcs = ["tpu_cluster_formation.cc"],
    textual_hdrs = [
        "clustering_passes.h.inc",
    ],
    deps = [
        ":clustering_passes_inc_gen",
        "//machina/compiler/mlir/machina",
        "//machina/compiler/mlir/machina:attribute_utils",
        "//machina/compiler/mlir/machina:string_util",
        "//machina/compiler/mlir/machina:machina_analysis",
        "//machina/compiler/mlir/machina:tpu_rewrite_device_util",
        "//machina/compiler/mlir/machina/transforms:tf_pass_inc_gen",
        "//machina/core:framework",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
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
    name = "extract_outside_compilation",
    srcs = ["extract_outside_compilation.cc"],
    textual_hdrs = [
        "clustering_passes.h.inc",
    ],
    deps = [
        ":clustering_passes_inc_gen",
        "//machina/compiler/mlir/machina",
        "//machina/compiler/mlir/machina:attribute_utils",
        "//machina/compiler/mlir/machina:device_util",
        "//machina/compiler/mlir/machina:serialize_mlir_module_utils",
        "//machina/compiler/mlir/machina:string_util",
        "//machina/compiler/mlir/machina:machina_analysis",
        "//machina/compiler/mlir/machina:machina_ops",
        "//machina/compiler/mlir/machina:machina_types",
        "//machina/compiler/mlir/machina:tpu_rewrite_device_util",
        "//machina/compiler/mlir/machina/transforms:shape_inference_pass",
        "//machina/core:framework",
        "//machina/core:lib",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:Rewrite",
        "@toolchain-project//mlir:Support",
        "@toolchain-project//mlir:TransformUtils",
    ],
)

cc_library(
    name = "extract_head_tail_outside_compilation",
    srcs = ["extract_head_tail_outside_compilation.cc"],
    textual_hdrs = [
        "clustering_passes.h.inc",
    ],
    deps = [
        ":clustering_passes_inc_gen",
        "//machina/compiler/mlir/machina",
        "//machina/compiler/mlir/machina:attribute_utils",
        "//machina/compiler/mlir/machina:device_util",
        "//machina/compiler/mlir/machina:string_util",
        "//machina/compiler/mlir/machina:machina_analysis",
        "//machina/compiler/mlir/machina:machina_ops",
        "//machina/compiler/mlir/machina:machina_types",
        "//machina/compiler/mlir/machina:tpu_rewrite_device_util",
        "//machina/core:framework",
        "//machina/core:lib",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:Rewrite",
        "@toolchain-project//mlir:Support",
        "@toolchain-project//mlir:TransformUtils",
    ],
)

cc_library(
    name = "mlir_to_graph_passes",
    hdrs = [
        "mlir_to_graph_passes.h",
    ],
    textual_hdrs = [
        "mlir_to_graph_passes.h.inc",
    ],
    deps = [
        ":verify_input_dialect_to_executor_pass",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:Pass",
    ],
)

gentbl_cc_library(
    name = "mlir_to_graph_passes_inc_gen",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = {"mlir_to_graph_passes.h.inc": [
        "-gen-pass-decls",
        "-name=TFXLABridgeMlirToGraph",
    ]},
    tblgen = "@toolchain-project//mlir:mlir-tblgen",
    td_file = "mlir_to_graph_passes.td",
    deps = [
        "@toolchain-project//mlir:PassBaseTdFiles",
    ],
)

cc_library(
    name = "verify_input_dialect_to_executor_pass",
    srcs = [
        "verify_input_dialect_to_executor_pass.cc",
    ],
    deps = [
        ":mlir_to_graph_passes_inc_gen",
        "//machina/compiler/mlir/tf2xla/internal/utils:dialect_detection_utils",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Pass",
    ],
)

cc_library(
    name = "xla_cluster_formation",
    srcs = ["xla_cluster_formation.cc"],
    textual_hdrs = [
        "clustering_passes.h.inc",
    ],
    deps = [
        ":clustering_passes_inc_gen",
        "//machina/compiler/mlir/machina",
        "//machina/compiler/mlir/machina:attribute_utils",
        "//machina/compiler/mlir/machina:call_graph_util",
        "//machina/compiler/mlir/machina:cluster_util",
        "//machina/compiler/mlir/machina:string_util",
        "//machina/compiler/mlir/machina:machina_analysis",
        "//machina/compiler/mlir/machina:machina_types",
        "//machina/compiler/mlir/machina:tpu_rewrite_device_util",
        "//machina/core:core_cpu_base",
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
    name = "mark_ops_for_outside_compilation",
    srcs = ["mark_ops_for_outside_compilation.cc"],
    textual_hdrs = [
        "clustering_passes.h.inc",
    ],
    deps = [
        ":clustering_passes_inc_gen",
        "//machina/compiler/mlir/machina",
        "//machina/compiler/mlir/machina:attribute_utils",
        "//machina/compiler/mlir/machina:string_util",
        "//machina/compiler/mlir/machina:machina_analysis",
        "//machina/compiler/mlir/machina:machina_ops",
        "//machina/compiler/mlir/machina:machina_types",
        "//machina/compiler/mlir/machina:tpu_rewrite_device_util",
        "//machina/compiler/mlir/machina/transforms:lower_tf_lib",
        "//machina/compiler/mlir/machina/transforms:tf_pass_inc_gen",
        "//machina/compiler/mlir/machina/transforms:verify_no_outside_compilation_markers_pass",
        "//machina/compiler/mlir/tf2xla/transforms:legalization_op_config",
        "//machina/compiler/mlir/tf2xla/transforms:legalize_tf",
        "//machina/core:framework",
        "//machina/core:lib",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:Rewrite",
        "@toolchain-project//mlir:Support",
        "@toolchain-project//mlir:TransformUtils",
    ],
)

cc_library(
    name = "tpu_sharding_identification_pass",
    srcs = ["tpu_sharding_identification_pass.cc"],
    textual_hdrs = [
        "clustering_passes.h.inc",
    ],
    deps = [
        ":clustering_passes_inc_gen",
        "//machina/compiler/mlir/machina",
        "//machina/compiler/mlir/machina:attribute_utils",
        "//machina/compiler/mlir/machina:string_util",
        "//machina/compiler/mlir/machina:machina_analysis",
        "//machina/compiler/mlir/machina:machina_ops",
        "//machina/compiler/mlir/machina:machina_traits",
        "//machina/compiler/mlir/machina:machina_types",
        "//machina/compiler/mlir/machina:tpu_rewrite_device_util",
        "//machina/compiler/mlir/machina:xla_sharding_util",
        "//machina/compiler/mlir/machina/transforms:lower_tf_lib",
        "//machina/compiler/mlir/machina/transforms:tf_pass_inc_gen",
        "//machina/compiler/mlir/machina/transforms:verify_no_outside_compilation_markers_pass",
        "//machina/compiler/mlir/tf2xla/transforms:legalization_op_config",
        "//machina/compiler/mlir/tf2xla/transforms:legalize_tf",
        "//machina/core:framework",
        "//machina/core:lib",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:Rewrite",
        "@toolchain-project//mlir:Support",
        "@toolchain-project//mlir:TransformUtils",
        "@local_xla//xla:xla_data_proto_cc",
        "@local_xla//xla/hlo/builder:sharding_builder",
    ],
)

cc_library(
    name = "hoist_broadcast_read",
    srcs = ["hoist_broadcast_read.cc"],
    textual_hdrs = [
        "clustering_passes.h.inc",
    ],
    deps = [
        ":clustering_passes_inc_gen",
        "//machina/compiler/mlir/machina",
        "//machina/compiler/mlir/machina:attribute_utils",
        "//machina/compiler/mlir/machina:string_util",
        "//machina/compiler/mlir/machina:machina_analysis",
        "//machina/compiler/mlir/machina:machina_ops",
        "//machina/compiler/mlir/machina:machina_types",
        "//machina/compiler/mlir/machina:tpu_rewrite_device_util",
        "//machina/compiler/mlir/machina/transforms:lower_tf_lib",
        "//machina/compiler/mlir/machina/transforms:tf_pass_inc_gen",
        "//machina/compiler/mlir/machina/transforms:verify_no_outside_compilation_markers_pass",
        "//machina/compiler/mlir/tf2xla/transforms:legalization_op_config",
        "//machina/compiler/mlir/tf2xla/transforms:legalize_tf",
        "//machina/core:framework",
        "//machina/core:lib",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:Rewrite",
        "@toolchain-project//mlir:Support",
        "@toolchain-project//mlir:TransformUtils",
    ],
)

cc_library(
    name = "xla_broadcast",
    srcs = ["xla_broadcast.cc"],
    textual_hdrs = [
        "clustering_passes.h.inc",
    ],
    deps = [
        ":clustering_passes_inc_gen",
        "//machina/compiler/mlir/machina",
        "//machina/compiler/mlir/machina:attribute_utils",
        "//machina/compiler/mlir/machina:device_util",
        "//machina/compiler/mlir/machina:string_util",
        "//machina/compiler/mlir/machina:machina_analysis",
        "//machina/compiler/mlir/machina:machina_ops",
        "//machina/compiler/mlir/machina:machina_types",
        "//machina/compiler/mlir/machina:tpu_rewrite_device_util",
        "//machina/compiler/mlir/machina:xla_rewrite_util",
        "//machina/compiler/mlir/machina:xla_sharding_util",
        "//machina/compiler/mlir/machina/transforms:lower_tf_lib",
        "//machina/compiler/mlir/machina/transforms:tf_pass_inc_gen",
        "//machina/compiler/mlir/machina/transforms:verify_no_outside_compilation_markers_pass",
        "//machina/compiler/mlir/tf2xla/transforms:legalization_op_config",
        "//machina/compiler/mlir/tf2xla/transforms:legalize_tf",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core/ir/types:Dialect",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:Rewrite",
        "@toolchain-project//mlir:Support",
        "@toolchain-project//mlir:TransformUtils",
    ],
)

cc_library(
    name = "lowering_passes",
    hdrs = [
        "lowering_passes.h",
    ],
    textual_hdrs = [
        "lowering_passes.h.inc",
    ],
    deps = [
        ":input_metrics_lowering_pass",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Pass",
    ],
)

gentbl_cc_library(
    name = "lowering_passes_inc_gen",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = {"lowering_passes.h.inc": [
        "-gen-pass-decls",
        "-name=TFXLABridgeLowering",
    ]},
    tblgen = "@toolchain-project//mlir:mlir-tblgen",
    td_file = "lowering_passes.td",
    deps = [
        "@toolchain-project//mlir:PassBaseTdFiles",
    ],
)

cc_library(
    name = "input_metrics_lowering_pass",
    srcs = [
        "input_lowering_metrics_pass.cc",
    ],
    deps = [
        ":lowering_passes_inc_gen",
        "//machina/compiler/mlir/machina",
        "//machina/compiler/mlir/tf2xla/transforms:legalization_op_config",
        "//machina/core:lib",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:Support",
    ],
)

tf_cc_test(
    name = "input_metrics_lowering_pass_test",
    srcs = ["input_lowering_metrics_pass_test.cc"],
    deps = [
        ":lowering_passes",
        "//machina/compiler/mlir/tf2xla/transforms:test_utils",
        "//machina/core/lib/monitoring:cell_reader",
        "@com_google_googletest//:gtest_main",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:Support",
        "@local_xla//xla/tsl/platform:statusor",
    ],
)

cc_library(
    name = "tpu_validate_inputs_utils",
    srcs = ["tpu_validate_inputs_utils.cc"],
    hdrs = ["tpu_validate_inputs_utils.h"],
    deps = [
        "//machina/compiler/mlir/machina",
        "//machina/compiler/mlir/machina:attribute_utils",
        "@com_google_absl//absl/strings",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Support",
    ],
)

cc_library(
    name = "tpu_validate_session_inputs",
    srcs = ["tpu_validate_session_inputs.cc"],
    textual_hdrs = [
        "clustering_passes.h.inc",
    ],
    deps = [
        ":clustering_passes_inc_gen",
        ":tpu_validate_inputs_utils",
        "//machina/compiler/mlir/machina",
        "//machina/compiler/mlir/machina:attribute_utils",
        "//machina/compiler/mlir/machina:dump_mlir_util",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Pass",
    ],
)

cc_library(
    name = "tpu_validate_inputs",
    srcs = ["tpu_validate_inputs.cc"],
    textual_hdrs = [
        "clustering_passes.h.inc",
    ],
    deps = [
        ":clustering_passes_inc_gen",
        ":tpu_validate_inputs_utils",
        "//machina/compiler/mlir/machina",
        "//machina/compiler/mlir/machina:attribute_utils",
        "//machina/compiler/mlir/machina:dump_mlir_util",
        "//machina/compiler/mlir/machina:string_util",
        "//machina/compiler/mlir/machina:machina_analysis",
        "//machina/compiler/mlir/machina:machina_types",
        "//machina/compiler/mlir/machina:tpu_rewrite_device_util",
        "//machina/compiler/mlir/machina/transforms:tf_pass_inc_gen",
        "//machina/core:framework",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:Support",
        "@toolchain-project//mlir:TransformUtils",
        "@local_xla//xla:xla_data_proto_cc",
        "@local_xla//xla/hlo/builder:sharding_builder",
    ],
)

tf_cc_test(
    name = "tpu_validate_inputs_utils_test",
    srcs = ["tpu_validate_inputs_utils_test.cc"],
    deps = [
        ":tpu_validate_inputs_utils",
        "//machina/compiler/mlir/machina",
        "//machina/compiler/mlir/tf2xla/transforms:test_utils",
        "@com_google_googletest//:gtest_main",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:IR",
    ],
)

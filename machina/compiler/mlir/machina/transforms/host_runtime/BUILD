load("@toolchain-project//mlir:tblgen.bzl", "gentbl_cc_library")
load("//machina:machina.bzl", "tf_cc_test")
load("//machina:machina.default.bzl", "get_compatible_with_portable")
load("//machina/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = [
        "//machina/compiler/mlir:__subpackages__",
        "//machina/compiler/mlir/machina/transforms:__pkg__",
        "//machina/compiler/mlir/tf2xla/api:__subpackages__",
        "//machina/compiler/mlir/tfrt:__subpackages__",
        "//machina/compiler/tf2xla:__pkg__",
        "//machina/dtensor/mlir:__pkg__",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "lower_cluster_to_runtime_ops",
    srcs = ["lower_cluster_to_runtime_ops.cc"],
    hdrs = [
        "lower_cluster_to_runtime_ops.h",
    ],
    deps = [
        ":runtime_passes",
        "//machina/compiler/jit:flags_headers",
        "//machina/compiler/mlir/machina:attribute_utils",
        "//machina/compiler/mlir/machina:bridge_logger",
        "//machina/compiler/mlir/machina:dump_mlir_util",
        "//machina/compiler/mlir/machina:error_util",
        "//machina/compiler/mlir/machina/transforms:verify_no_outside_compilation_markers_pass",
        "//machina/compiler/mlir/machina/transforms/sparsecore:sparsecore_passes",
        "//machina/core:framework",
        "//machina/core:lib_proto_parsing",
        "//machina/core/platform:error_payloads",
        "//machina/core/platform:status",
        "//machina/core/tpu:tpu_defs",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:Support",
        "@toolchain-project//mlir:Transforms",
        "@local_tsl//tsl/platform:error_logging",
        "@local_xla//xla/tsl/platform:errors",
    ],
)

tf_cc_test(
    name = "lower_cluster_to_runtime_ops_test",
    srcs = ["lower_cluster_to_runtime_ops_test.cc"],
    data = [
        "testdata/basic_cluster.mlir",
        "testdata/empty_func.mlir",
        "testdata/malformed_cluster.mlir",
    ],
    deps = [
        ":lower_cluster_to_runtime_ops",
        "//machina/compiler/mlir:register_common_dialects",
        "//machina/compiler/mlir/machina",
        "//machina/compiler/mlir/machina:attribute_utils",
        "//machina/compiler/tf2xla:xla_op_registry",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:test",
        "//machina/core/lib/monitoring:cell_reader",
        "//machina/core/platform:resource_loader",
        "//machina/core/tpu:tpu_defs",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Parser",
        "@toolchain-project//mlir:Pass",
    ],
)

cc_library(
    name = "tpu_metadata_utils",
    srcs = [
        "tpu_metadata_utils.cc",
    ],
    hdrs = [
        "tpu_metadata_utils.h",
    ],
    deps = [
        "//machina/compiler/mlir/machina",
        "//machina/compiler/mlir/machina:attribute_utils",
        "//machina/compiler/mlir/machina:convert_tensor",
        "//machina/compiler/mlir/machina:convert_type",
        "//machina/compiler/mlir/machina:xla_sharding_util",
        "//machina/core:core_cpu_base",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core/protobuf/tpu:compile_metadata_proto_cc",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Support",
        "@local_xla//xla:xla_data_proto_cc",
        "@local_xla//xla:xla_proto_cc",
    ],
)

tf_cc_test(
    name = "tpu_metadata_utils_test",
    srcs = ["tpu_metadata_utils_test.cc"],
    data = [
        "testdata/basic_cluster.mlir",
        "testdata/spmd.mlir",
    ],
    deps = [
        ":tpu_metadata_utils",
        "//machina/compiler/mlir:register_common_dialects",
        "//machina/compiler/mlir/machina",
        "//machina/compiler/tf2xla:xla_op_registry",
        "//machina/core:lib",
        "//machina/core:test",
        "//machina/core/platform:resource_loader",
        "//machina/core/protobuf/tpu:compile_metadata_proto_cc",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Parser",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:Support",
    ],
)

gentbl_cc_library(
    name = "runtime_passes_inc_gen",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = {"runtime_passes.h.inc": [
        "-gen-pass-decls",
        "-name=RuntimeLowering",
    ]},
    tblgen = "@toolchain-project//mlir:mlir-tblgen",
    td_file = "runtime_passes.td",
    deps = [
        "@toolchain-project//mlir:PassBaseTdFiles",
    ],
)

cc_library(
    name = "runtime_passes",
    hdrs = [
        "runtime_passes.h",
    ],
    textual_hdrs = [
        "runtime_passes.h.inc",
    ],
    deps = [
        ":runtime_passes_inc_gen",
        ":tpu_merge_variable_with_execute",
        ":tpu_rewrite_pass",
        ":tpu_variable_runtime_reformatting",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Pass",
    ],
)

cc_library(
    name = "tpu_rewrite_pass",
    srcs = ["tpu_rewrite_pass.cc"],
    hdrs = [
        "runtime_passes.h",
    ],
    deps = [
        ":runtime_passes_inc_gen",
        ":tpu_metadata_utils",
        "//machina/compiler/jit:flags_headers",
        "//machina/compiler/mlir/machina",
        "//machina/compiler/mlir/machina:attribute_utils",
        "//machina/compiler/mlir/machina:bridge_logger",
        "//machina/compiler/mlir/machina:convert_tensor",
        "//machina/compiler/mlir/machina:convert_type",
        "//machina/compiler/mlir/machina:device_util",
        "//machina/compiler/mlir/machina:dump_mlir_util",
        "//machina/compiler/mlir/machina:dynamic_shape_utils",
        "//machina/compiler/mlir/machina:error_util",
        "//machina/compiler/mlir/machina:parallel_execute_util",
        "//machina/compiler/mlir/machina:serialize_mlir_module_utils",
        "//machina/compiler/mlir/machina:machina_types",
        "//machina/compiler/mlir/machina:tpu_rewrite_device_util",
        "//machina/compiler/mlir/machina:xla_rewrite_util",
        "//machina/compiler/mlir/machina:xla_sharding_util",
        "//machina/core:framework",
        "//machina/core:lib_proto_parsing",
        "//machina/core:protos_all_cc",
        "//machina/core/platform:error_payloads",
        "//machina/core/platform:fingerprint",
        "//machina/core/platform:status",
        "//machina/core/protobuf/tpu:compile_metadata_proto_cc",
        "//machina/core/tpu:tpu_defs",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:Support",
        "@toolchain-project//mlir:Transforms",
        "@local_tsl//tsl/platform:error_logging",
        "@local_xla//xla:xla_data_proto_cc",
        "@local_xla//xla:xla_proto_cc",
        "@local_xla//xla/mlir_hlo",
        "@local_xla//xla/tsl/platform:errors",
    ],
)

cc_library(
    name = "tpu_variable_runtime_reformatting",
    srcs = ["tpu_variable_runtime_reformatting.cc"],
    hdrs = [
        "runtime_passes.h",
    ],
    deps = [
        ":runtime_passes_inc_gen",
        "//machina/compiler/mlir/machina",
        "//machina/compiler/mlir/machina:convert_tensor",
        "//machina/compiler/mlir/machina:mangling_util",
        "//machina/compiler/mlir/machina:machina_types",
        "//machina/compiler/mlir/machina:tpu_rewrite_device_util",
        "//machina/core:framework",
        "//machina/core:protos_all_cc",
        "//machina/core/platform:random",
        "//machina/core/protobuf/tpu:compile_metadata_proto_cc",
        "@com_google_absl//absl/strings",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:Support",
        "@toolchain-project//mlir:Transforms",
        "@local_xla//xla/mlir_hlo",
    ],
)

cc_library(
    name = "tpu_merge_variable_with_execute",
    srcs = ["tpu_merge_variables_with_execute.cc"],
    hdrs = [
        "runtime_passes.h",
    ],
    deps = [
        ":runtime_passes_inc_gen",
        "//machina/compiler/mlir/machina",
        "//machina/compiler/mlir/machina:machina_analysis",
        "//machina/compiler/mlir/machina:machina_ops",
        "//machina/compiler/mlir/machina:machina_types",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:Support",
        "@local_xla//xla/mlir_hlo",
    ],
)

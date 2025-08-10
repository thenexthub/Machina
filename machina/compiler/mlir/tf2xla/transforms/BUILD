# Description:
#    TF2XLA Bridge transforms

load("@toolchain-project//mlir:tblgen.bzl", "gentbl_cc_library")
load("@local_xla//xla/tsl/platform:build_config_root.bzl", "if_static")
load("//machina:machina.bzl", "tf_cc_test")
load("//machina:machina.default.bzl", "get_compatible_with_portable")
load("//machina/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

gentbl_cc_library(
    name = "legalize_tf_patterns_inc_gen",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = {"generated_legalize_tf.inc": ["-gen-rewriters"]},
    tblgen = "@toolchain-project//mlir:mlir-tblgen",
    td_file = "legalize_tf_patterns.td",
    deps = [
        "//machina/compiler/mlir/machina:machina_ops_td_files",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:FuncTdFiles",
        "@toolchain-project//mlir:TensorOpsTdFiles",
        "@local_xla//xla/mlir_hlo:hlo_ops_td_files",
        "@stablehlo//:chlo_ops_td_files",
        "@stablehlo//:stablehlo_ops_td_files",
    ],
)

gentbl_cc_library(
    name = "xla_legalize_tf_passes_inc_gen",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = {"xla_legalize_tf_passes.h.inc": [
        "-gen-pass-decls",
        "-name=LegalizeTf",
    ]},
    tblgen = "@toolchain-project//mlir:mlir-tblgen",
    td_file = "xla_legalize_tf_passes.td",
    deps = [
        "@toolchain-project//mlir:PassBaseTdFiles",
    ],
)

gentbl_cc_library(
    name = "tf_xla_passes_inc_gen",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = {"tf_xla_passes.h.inc": [
        "-gen-pass-decls",
        "-name=TfXla",
    ]},
    tblgen = "@toolchain-project//mlir:mlir-tblgen",
    td_file = "tf_xla_passes.td",
    deps = [
        "//machina/compiler/mlir/machina:machina_ops_td_files",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:FuncTdFiles",
        "@toolchain-project//mlir:PassBaseTdFiles",
        "@toolchain-project//mlir:SparseTensorDialect",
        "@toolchain-project//mlir:TensorOpsTdFiles",
        "@local_xla//xla/mlir_hlo:hlo_ops_td_files",
    ],
)

cc_library(
    name = "tf_xla_passes",
    srcs = [
        "xla_legalize_tf_passes.h.inc",
    ],
    hdrs = [
        "passes.h",
    ],
    deps = [
        ":tf_xla_passes_inc_gen",
        ":xla_legalize_tf",
        ":xla_legalize_tf_passes_inc_gen",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:MemRefDialect",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:SparseTensorDialect",
        "@toolchain-project//mlir:Support",
        "@toolchain-project//mlir:TransformUtils",
        "@local_xla//xla/mlir_hlo",
    ],
)

cc_library(
    name = "legalize_utils",
    srcs = ["utils.cc"],
    hdrs = ["utils.h"],
    deps = [
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:IR",
        "@local_xla//xla/mlir_hlo",
    ],
)

cc_library(
    name = "test_utils",
    testonly = True,
    srcs = ["test_utils.cc"],
    hdrs = ["test_utils.h"],
    deps = [
        "//machina/compiler/mlir:register_common_dialects",
        "//machina/compiler/mlir/machina",
        "//machina/compiler/mlir/machina:deserialize_mlir_module_utils",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:FuncDialect",
        "@local_xla//xla/tsl/platform:statusor",
    ],
)

cc_library(
    name = "legalize_tf",
    srcs = [
        "generated_legalize_tf.inc",
        "legalize_tf.cc",
    ],
    hdrs = [
        "passes.h",
    ],
    # DEPRECATED: use v2/legalize_tf.h::LegalizeMlirToHlo instead.
    visibility = [
        "//machina/compiler/mlir/machina/transforms:__pkg__",
        "//machina/compiler/mlir/tf2xla/internal/passes:__pkg__",
    ],
    deps = [
        ":legalize_tf_patterns_inc_gen",
        ":legalize_utils",
        ":tf_xla_passes_inc_gen",
        ":xla_legalize_tf_passes_inc_gen",
        "//machina/compiler/mlir/machina",
        "//machina/compiler/mlir/machina:dynamic_shape_utils",
        "//machina/compiler/mlir/machina:xla_sharding_util",
        "//machina/core:framework",
        "//machina/core/kernels:conv_grad_shape_utils",
        "@com_google_absl//absl/status",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:ArithDialect",
        "@toolchain-project//mlir:Dialect",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:MemRefDialect",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:ShapeDialect",
        "@toolchain-project//mlir:Support",
        "@toolchain-project//mlir:TensorDialect",
        "@toolchain-project//mlir:TransformUtils",
        "@local_tsl//tsl/platform:bfloat16",
        "@local_tsl//tsl/platform:tensor_float_32_hdr_lib",
        "@local_xla//xla:xla_data_proto_cc",
        "@local_xla//xla/hlo/builder:padding",
        "@local_xla//xla/hlo/builder:sharding_builder",
        "@local_xla//xla/hlo/builder/lib:conv_grad_size_util",
        "@local_xla//xla/hlo/translate/hlo_to_mhlo:attribute_importer",
        "@local_xla//xla/mlir_hlo",
        "@local_xla//xla/mlir_hlo:convert_op_folder",
        "@local_xla//xla/tsl/platform:status",
        "@stablehlo//:chlo_ops",
        "@stablehlo//:stablehlo_ops",
        "@stablehlo//:stablehlo_pass_utils",
    ] + if_static(["@local_tsl//tsl/platform:tensor_float_32_utils"]),
)

cc_library(
    name = "xla_legalize_targets",
    srcs = [
        "xla_legalize_targets.cc",
    ],
    hdrs = [
        "xla_legalize_targets.h",
    ],
    deps = [
        "//machina/compiler/mlir/machina",
        "@toolchain-project//mlir:ArithDialect",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:ShapeDialect",
        "@toolchain-project//mlir:TensorDialect",
        "@toolchain-project//mlir:TransformUtils",
        "@local_xla//xla/mlir_hlo",
        "@stablehlo//:chlo_ops",
        "@stablehlo//:stablehlo_ops",
    ],
)

tf_cc_test(
    name = "xla_legalize_targets_test",
    srcs = ["xla_legalize_targets_test.cc"],
    deps = [
        ":xla_legalize_targets",
        "//machina/compiler/mlir/machina",
        "@com_google_googletest//:gtest_main",
        "@toolchain-project//mlir:ArithDialect",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:ShapeDialect",
        "@toolchain-project//mlir:TensorDialect",
        "@toolchain-project//mlir:TransformUtils",
        "@stablehlo//:chlo_ops",
    ],
)

tf_cc_test(
    name = "verify_tfxla_legalization_test",
    srcs = ["verify_tfxla_legalization_test.cc"],
    deps = [
        ":legalize_tf",
        ":test_utils",
        "//machina/compiler/mlir/machina",
        "//machina/compiler/mlir/machina:serialize_mlir_module_utils",
        "//machina/core:test",
        "//machina/core/lib/monitoring:cell_reader",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:Support",
        "@local_xla//xla/tsl/platform:statusor",
    ],
)

cc_library(
    name = "xla_legalize_tf",
    srcs = [
        "infeed_ops_xla_adjust_layout.cc",
        "legalize_tf_collective.cc",
        "legalize_tf_communication.cc",
        "tf_xla_passes.h.inc",
        "tfxla_device_specific_transforms.cc",
        "verify_tfxla_legalization.cc",
        "xla_legalize_tf.cc",
        "xla_legalize_tf_passes.h.inc",
    ],
    hdrs = [
        "passes.h",
    ],
    deps = [
        ":legalization_op_config",
        ":legalize_tf",
        ":legalize_utils",
        ":tf_xla_passes_inc_gen",
        ":xla_legalize_targets",
        ":xla_legalize_tf_passes_inc_gen",
        ":xla_legalize_tf_with_tf2xla",
        "//machina/compiler/mlir/machina",
        "//machina/compiler/mlir/machina:mangling_util",
        "//machina/compiler/mlir/machina:machina_types",
        "//machina/compiler/mlir/machina/transforms:lower_tf_lib",
        "//machina/compiler/mlir/machina/transforms:set_tpu_infeed_layout",
        "//machina/compiler/tf2xla/kernels:rng_converter_utils",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core/util/quantization:uniform_quant_ops_params",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:ArithDialect",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:FuncExtensions",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:MemRefDialect",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:QuantOps",
        "@toolchain-project//mlir:ShapeDialect",
        "@toolchain-project//mlir:SparseTensorDialect",
        "@toolchain-project//mlir:Support",
        "@toolchain-project//mlir:TensorDialect",
        "@toolchain-project//mlir:TransformUtils",
        "@toolchain-project//mlir:Transforms",
        "@local_xla//xla:shape_util",
        "@local_xla//xla:side_effect_util",
        "@local_xla//xla:xla_data_proto_cc",
        "@local_xla//xla/hlo/builder:padding",
        "@local_xla//xla/hlo/builder:sharding_builder",
        "@local_xla//xla/hlo/translate/hlo_to_mhlo:attribute_importer",
        "@local_xla//xla/hlo/translate/mhlo_to_hlo:type_to_shape",
        "@local_xla//xla/mlir_hlo",
        "@local_xla//xla/mlir_hlo:convert_op_folder",
        "@local_xla//xla/mlir_hlo:mhlo_passes",
        "@local_xla//xla/mlir_hlo:type_conversion",
        "@local_xla//xla/stream_executor/tpu:c_api_conversions",
        "@local_xla//xla/stream_executor/tpu:tpu_api",
        "@stablehlo//:base",
        "@stablehlo//:chlo_ops",
        "@stablehlo//:stablehlo_ops",
        "@stablehlo//:stablehlo_passes",
    ],
)

cc_library(
    name = "tf2xla_rewriter",
    srcs = [
        "tf2xla_rewriter.cc",
    ],
    hdrs = [
        "tf2xla_rewriter.h",
    ],
    visibility = ["//visibility:private"],
    deps = [
        ":legalize_tf",
        "//machina/compiler/mlir:op_or_arg_name_mapper",
        "//machina/compiler/mlir/machina",
        "//machina/compiler/mlir/machina:convert_tensor",
        "//machina/compiler/mlir/machina:convert_type",
        "//machina/compiler/mlir/machina:tpu_embedding_ops_registry",
        "//machina/compiler/mlir/machina:translate_utils",
        "//machina/compiler/tf2xla:xla_compilation_device",
        "//machina/compiler/tf2xla:xla_context",
        "//machina/compiler/tf2xla:xla_expression",
        "//machina/compiler/tf2xla:xla_helpers",
        "//machina/compiler/tf2xla:xla_op_registry",
        "//machina/core:core_cpu",
        "//machina/core:core_cpu_base",
        "//machina/core:framework",
        "//machina/core:framework_types_hdr",
        "//machina/core:protos_all_cc",
        "//machina/core/framework:allocator",
        "//machina/core/protobuf:for_core_protos_cc",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:SparseTensorDialect",
        "@toolchain-project//mlir:Support",
        "@toolchain-project//mlir:TensorDialect",
        "@toolchain-project//mlir:TransformUtils",
        "@local_xla//xla:xla_data_proto_cc",
        "@local_xla//xla:xla_proto_cc",
        "@local_xla//xla/hlo/builder:xla_builder",
        "@local_xla//xla/hlo/builder:xla_computation",
        "@local_xla//xla/hlo/ir:hlo",
        "@local_xla//xla/hlo/translate/hlo_to_mhlo:hlo_function_importer",
        "@local_xla//xla/hlo/translate/hlo_to_mhlo:hlo_to_mlir_hlo",
        "@local_xla//xla/hlo/translate/mhlo_to_hlo:type_to_shape",
        "@local_xla//xla/service:hlo_proto_cc",
        "@local_xla//xla/tsl/platform:env",
        "@local_xla//xla/tsl/platform:errors",
        "@local_xla//xla/tsl/platform:status",
        "@local_xla//xla/tsl/platform:statusor",
        "@stablehlo//:base",
        "@stablehlo//:stablehlo_ops",
    ],
)

tf_cc_test(
    name = "tf2xla_rewriter_test",
    srcs = [
        "tf2xla_rewriter_test.cc",
    ],
    deps = [
        ":test_utils",
        ":tf2xla_rewriter",
        "//machina/compiler/mlir/machina",
        "//machina/compiler/tf2xla:xla_op_registry",
        "//machina/compiler/tf2xla/kernels:xla_ops",
        "//machina/core:ops",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest_main",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Support",
        "@toolchain-project//mlir:TransformUtils",
        "@local_xla//xla:shape_util",
        "@local_xla//xla:xla_data_proto_cc",
        "@local_xla//xla/hlo/builder:xla_builder",
        "@local_xla//xla/hlo/builder:xla_computation",
        "@local_xla//xla/tsl/lib/core:status_test_util",
        "@local_xla//xla/tsl/platform:errors",
        "@local_xla//xla/tsl/platform:status",
        "@local_xla//xla/tsl/platform:statusor",
        "@stablehlo//:stablehlo_ops",
    ],
)

cc_library(
    name = "xla_legalize_tf_with_tf2xla",
    srcs = [
        "legalize_tf_with_tf2xla.cc",
    ],
    hdrs = [
        "legalize_tf_with_tf2xla_passes.h",
        "passes.h",
    ],
    deps = [
        ":legalization_op_config",
        ":tf2xla_rewriter",
        ":tf_xla_passes_inc_gen",
        ":xla_legalize_tf_passes_inc_gen",
        "//machina/compiler/mlir:op_or_arg_name_mapper",
        "//machina/compiler/mlir/machina",
        "//machina/compiler/mlir/machina:machina_ops",
        "//machina/compiler/mlir/machina:tpu_embedding_ops_registry",
        "//machina/compiler/tf2xla:xla_compilation_device",
        "//machina/compiler/tf2xla:xla_context",
        "//machina/compiler/tf2xla:xla_expression",
        "//machina/compiler/tf2xla:xla_helpers",
        "//machina/compiler/tf2xla:xla_op_registry",
        "//machina/core:core_cpu_lib",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core:protos_all_cc",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:Support",
        "@toolchain-project//mlir:TensorDialect",
        "@toolchain-project//mlir:TransformUtils",
        "@stablehlo//:base",
    ],
)

tf_cc_test(
    name = "xla_legalize_tf_test",
    srcs = [
        "xla_legalize_tf_test.cc",
    ],
    deps = [
        ":tf_xla_passes",
        "//machina/compiler/mlir/machina",
        "//machina/compiler/mlir/machina:deserialize_mlir_module_utils",
        "//machina/compiler/tf2xla:xla_compilation_device",
        "//machina/compiler/tf2xla:xla_context",
        "//machina/compiler/tf2xla:xla_expression",
        "//machina/compiler/tf2xla:xla_helpers",
        "//machina/compiler/tf2xla:xla_op_registry",
        "//machina/core:core_cpu_base",
        "//machina/core/framework:allocator",
        "//machina/core/lib/monitoring:cell_reader",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:TransformUtils",
        "@local_xla//xla/tsl/platform:statusor",
    ],
)

cc_library(
    name = "legalization_op_config",
    srcs = ["legalization_op_config.cc"],
    hdrs = ["legalization_op_config.h"],
    visibility = [
        "//machina/compiler/mlir/machina/transforms:__pkg__",
        "//machina/compiler/mlir/tf2xla/internal:__subpackages__",
    ],
    deps = [
        "//machina/compiler/mlir/machina",
        "//machina/compiler/mlir/machina:tpu_embedding_ops_registry",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:Support",
    ],
)

tf_cc_test(
    name = "legalization_op_config_test",
    srcs = ["legalization_op_config_test.cc"],
    deps = [
        ":legalization_op_config",
        ":legalize_tf",
        "//machina/compiler/jit",
        "//machina/compiler/mlir:register_common_dialects",
        "//machina/compiler/mlir/machina",
        "//machina/compiler/tf2xla:xla_op_registry",
        "//machina/compiler/tf2xla/kernels:xla_ops",
        "//machina/core:protos_all_cc",
        "@com_google_googletest//:gtest_main",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Support",
    ],
)

cc_library(
    name = "split_into_island_per_op_pass",
    srcs = ["split_into_island_per_op_pass.cc"],
    hdrs = [
        "split_into_island_per_op_pass.h",
    ],
    deps = [
        "//machina/compiler/mlir/machina",
        "//machina/compiler/mlir/machina:machina_executor_inc_gen",
        "//machina/compiler/mlir/machina:machina_types",
        "//machina/compiler/mlir/machina/transforms:tf_pass_inc_gen",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:Dialect",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:InferTypeOpInterface",
        "@toolchain-project//mlir:Pass",
    ],
)

tf_cc_test(
    name = "split_into_island_per_op_pass_test",
    srcs = ["split_into_island_per_op_pass_test.cc"],
    deps = [
        ":split_into_island_per_op_pass",
        "//machina/compiler/mlir:register_common_dialects",
        "//machina/compiler/mlir/machina",
        "@com_google_googletest//:gtest_main",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Support",
    ],
)

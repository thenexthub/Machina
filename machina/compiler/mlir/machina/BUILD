# buildifier: disable=out-of-order-load

load("@rules_shell//shell:sh_test.bzl", "sh_test")
load("//machina:strict.default.bzl", "py_strict_library")

# copybara:uncomment_begin(google-only)
# load("//learning/brain/experimental/mlir/machina/dialectgen:dialectgen.bzl", "dialectgen")
#
# copybara:uncomment_end(google-only)
load("@bazel_skylib//rules:build_test.bzl", "build_test")
load("@toolchain-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")
load("//machina:machina.bzl", "tf_cc_test", "tf_gen_op_wrapper_py")
load("//machina:machina.default.bzl", "get_compatible_with_portable")
load("//machina/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

exports_files([
    "ir/tf_generated_ops.td",
    "ir/tf_op_base.td",
    "ir/tf_op_interfaces.td",
    "ir/tf_ops.td",
])

td_library(
    name = "machina_ops_td_files",
    srcs = [
        "ir/tf_generated_ops.td",
        "ir/tf_op_base.td",
        "ir/tf_op_interfaces.td",
        "ir/tf_ops.td",
    ],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "@toolchain-project//mlir:CallInterfacesTdFiles",
        "@toolchain-project//mlir:ControlFlowInterfacesTdFiles",
        "@toolchain-project//mlir:InferTypeOpInterfaceTdFiles",
        "@toolchain-project//mlir:LoopLikeInterfaceTdFiles",
        "@toolchain-project//mlir:OpBaseTdFiles",
        "@toolchain-project//mlir:SideEffectInterfacesTdFiles",
    ],
)

gentbl_cc_library(
    name = "machina_op_interfaces_inc_gen",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = {
        "ir/tf_op_interfaces.h.inc": ["-gen-op-interface-decls"],
        "ir/tf_op_interfaces.cc.inc": ["-gen-op-interface-defs"],
    },
    tblgen = "@toolchain-project//mlir:mlir-tblgen",
    td_file = "ir/tf_op_interfaces.td",
    test = True,
    deps = [
        ":machina_ops_td_files",
    ],
)

gentbl_cc_library(
    name = "machina_struct_doc_gen",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = {"g3doc/tf_ops.md": ["-gen-dialect-doc"]},
    tblgen = "@toolchain-project//mlir:mlir-tblgen",
    td_file = "ir/tf_ops.td",
    test = True,
    deps = [
        ":machina_ops_td_files",
    ],
)

cc_library(
    name = "machina_op_interfaces",
    srcs = [
        "ir/tf_op_interfaces.cc",
        "ir/tf_op_interfaces.cc.inc",
        "ir/tf_op_interfaces.h.inc",
        "ir/tf_verifiers.cc",
    ],
    hdrs = [
        "ir/tf_op_interfaces.h",
        "ir/tf_verifiers.h",
    ],
    deps = [
        ":machina_op_interfaces_inc_gen",
        ":machina_structs",
        "//machina/core:framework",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Support",
    ],
)

gentbl_cc_library(
    name = "machina_all_ops_inc_gen",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = {
        "ir/tf_all_ops.h.inc": ["-gen-op-decls"],
        "ir/tf_all_ops.cc.inc": ["-gen-op-defs"],
    },
    tblgen = "@toolchain-project//mlir:mlir-tblgen",
    td_file = "ir/tf_ops.td",
    deps = [
        ":machina_ops_td_files",
    ],
)

# We only shard tf_op on name for build performance reasons.
tf_ops_category_list = [
    {
        "name": "ops_a_m",
        "include": "tf.[A-M].*$",
    },
    {
        "name": "ops_n_z",
        "include": "tf.[N-Z].*$",
    },
]

[[
    gentbl_cc_library(
        name = "machina_" + target["name"] + "_inc_gen",
        compatible_with = get_compatible_with_portable(),
        tbl_outs = {
            "ir/tf_" + target["name"] + ".h.inc": [
                "-gen-op-decls",
                "-op-include-regex=" + target["include"],
            ],
            "ir/tf_" + target["name"] + ".cc.inc": [
                "-gen-op-defs",
                "-op-include-regex=" + target["include"],
            ],
        },
        tblgen = "@toolchain-project//mlir:mlir-tblgen",
        td_file = "ir/tf_ops.td",
        deps = [
            ":machina_ops_td_files",
        ],
    ),
] for target in tf_ops_category_list]

gentbl_cc_library(
    name = "machina_remaining_ops_inc_gen",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = {
        "ir/tf_remaining_ops.h.inc": [
            "-gen-op-decls",
            "-op-exclude-regex=" + "|".join([target["include"] for target in tf_ops_category_list]),
        ],
        "ir/tf_remaining_ops.cc.inc": [
            "-gen-op-defs",
            "-op-exclude-regex=" + "|".join([target["include"] for target in tf_ops_category_list]),
        ],
    },
    tblgen = "@toolchain-project//mlir:mlir-tblgen",
    td_file = "ir/tf_ops.td",
    deps = [
        ":machina_ops_td_files",
    ],
)

gentbl_cc_library(
    name = "tf_saved_model_inc_gen",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = {
        "ir/tf_saved_model.h.inc": ["-gen-op-decls"],
        "ir/tf_saved_model.cc.inc": ["-gen-op-defs"],
        "g3doc/tf_saved_model.md": ["-gen-dialect-doc"],
    },
    tblgen = "@toolchain-project//mlir:mlir-tblgen",
    td_file = "ir/tf_saved_model_ops.td",
    test = True,
    deps = [
        "@toolchain-project//mlir:FuncTdFiles",
        "@toolchain-project//mlir:OpBaseTdFiles",
    ],
)

gentbl_cc_library(
    name = "machina_executor_inc_gen",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = {
        "ir/tf_executor.h.inc": ["-gen-op-decls"],
        "ir/tf_executor.cc.inc": ["-gen-op-defs"],
        "g3doc/tf_executor.md": [
            "-gen-dialect-doc",
            "-dialect=tf_executor",
        ],
    },
    tblgen = "@toolchain-project//mlir:mlir-tblgen",
    td_file = "ir/tf_executor_ops.td",
    test = True,
    deps = [
        ":machina_ops_td_files",
        "@toolchain-project//mlir:FuncTdFiles",
        "@toolchain-project//mlir:InferTypeOpInterfaceTdFiles",
        "@toolchain-project//mlir:OpBaseTdFiles",
    ],
)

gentbl_cc_library(
    name = "machina_device_ops_inc_gen",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = {
        "ir/tf_device.h.inc": ["-gen-op-decls"],
        "ir/tf_device.cc.inc": ["-gen-op-defs"],
        "g3doc/tf_device.md": ["-gen-dialect-doc"],
    },
    tblgen = "@toolchain-project//mlir:mlir-tblgen",
    td_file = "ir/tf_device_ops.td",
    test = True,
    deps = [
        "@toolchain-project//mlir:ControlFlowInterfacesTdFiles",
        "@toolchain-project//mlir:FuncTdFiles",
        "@toolchain-project//mlir:OpBaseTdFiles",
        "@toolchain-project//mlir:SideEffectInterfacesTdFiles",
    ],
)

cc_library(
    name = "machina_attributes",
    hdrs = [
        "ir/tf_attributes.h",
        "ir/tf_dialect.h",
    ],
    deps = [
        ":machina_types",
        "//machina/core/ir/types:Dialect",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Parser",
        "@toolchain-project//mlir:Support",
    ],
)

cc_library(
    name = "machina_traits",
    srcs = [
    ],
    hdrs = [
        "ir/tf_traits.h",
    ],
    deps = [
        ":machina_op_interfaces",
        ":machina_types",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:InferTypeOpInterface",
        "@toolchain-project//mlir:SideEffectInterfaces",
        "@toolchain-project//mlir:Support",
    ],
)

# TensorFlow ops are separated into `machina_ops_a_m.cc` and
# `machina_ops_n_z.cc` so that C++ compiler won't be stressed by huge C++
# files. However, there might be dependencies between `machina_ops_a_m.cc`
# and `machina_ops_n_z.cc`, thus they must be built in one `cc_library`.
cc_library(
    name = "machina_ops_sharded",
    srcs = [
               "ir/tf_dialect.h",
               "ir/tf_ops.h",
               "ir/tf_remaining_ops.h",
           ] + ["ir/tf_" + target["name"] + ".cc" for target in tf_ops_category_list] +
           ["ir/tf_" + target["name"] + ".cc.inc" for target in tf_ops_category_list] +
           ["ir/tf_" + target["name"] + ".h" for target in tf_ops_category_list],
    hdrs = [
    ],
    textual_hdrs = [
        "ir/tf_types.def",
        "ir/tf_all_ops.h.inc",
        "ir/tf_remaining_ops.h.inc",
    ] + ["ir/tf_" + target["name"] + ".h.inc" for target in tf_ops_category_list],
    deps = [
        ":attribute_utils",
        ":convert_type",
        ":dynamic_shape_utils",
        ":side_effect_analysis_util",
        ":machina_all_ops_inc_gen",
        ":machina_attributes",
        ":machina_op_interfaces",
        ":machina_op_interfaces_inc_gen",
        ":machina_remaining_ops_inc_gen",
        ":machina_side_effects",
        ":machina_structs",
        ":machina_traits",
        ":machina_types",
        ":tf_arith_ops_folder",
        ":tf_ops_canonicalization_helper",
        ":tf_ops_device_helper",
        ":tf_ops_layout_helper",
        ":tf_ops_tensor_helper",
        "//machina/compiler/mlir/machina/ir/host_runtime:machina_tfrt_ops",
        "//machina/compiler/mlir/machina/ir/host_runtime:machina_tfrt_ops_inc_gen",
        "//machina/compiler/mlir/machina/transforms:rewrite_util",
        "//machina/compiler/mlir/machina/transforms:machina_canonicalize_inc_gen",
        "//machina/core:framework",
        "//machina/core:lib",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:BytecodeOpInterface",
        "@toolchain-project//mlir:CallOpInterfaces",
        "@toolchain-project//mlir:ControlFlowInterfaces",
        "@toolchain-project//mlir:DerivedAttributeOpInterface",
        "@toolchain-project//mlir:Dialect",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:InferTypeOpInterface",
        "@toolchain-project//mlir:InliningUtils",
        "@toolchain-project//mlir:LoopLikeInterface",
        "@toolchain-project//mlir:Parser",
        "@toolchain-project//mlir:SideEffectInterfaces",
        "@toolchain-project//mlir:Support",
    ] + [":machina_" + target["name"] + "_inc_gen" for target in tf_ops_category_list],
)

cc_library(
    name = "machina_remaining_ops",
    srcs = [
        "ir/tf_dialect.h",
        "ir/tf_ops.h",
        "ir/tf_remaining_ops.cc",
        "ir/tf_remaining_ops.h",
    ] + ["ir/tf_" + target["name"] + ".h" for target in tf_ops_category_list],
    hdrs = [
    ],
    textual_hdrs = [
        "ir/tf_all_ops.h.inc",
        "ir/tf_remaining_ops.h.inc",
    ] + ["ir/tf_" + target["name"] + ".h.inc" for target in tf_ops_category_list],
    deps = [
        ":attribute_utils",
        ":deserialize_mlir_module_utils",
        ":side_effect_analysis_util",
        ":machina_attributes",
        ":machina_op_interfaces",
        ":machina_op_interfaces_inc_gen",
        ":machina_remaining_ops_inc_gen",
        ":machina_side_effects",
        ":machina_structs",
        ":machina_traits",
        ":machina_types",
        "//machina/compiler/mlir/machina/ir/host_runtime:machina_tfrt_ops",
        "//machina/compiler/mlir/machina/ir/host_runtime:machina_tfrt_ops_inc_gen",
        "//machina/compiler/mlir/machina/transforms:rewrite_util",
        "//machina/compiler/mlir/machina/transforms:machina_canonicalize_inc_gen",
        "//machina/core:framework",
        "//machina/core:lib",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:BytecodeOpInterface",
        "@toolchain-project//mlir:CallOpInterfaces",
        "@toolchain-project//mlir:ControlFlowInterfaces",
        "@toolchain-project//mlir:DerivedAttributeOpInterface",
        "@toolchain-project//mlir:Dialect",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:InferTypeOpInterface",
        "@toolchain-project//mlir:InliningUtils",
        "@toolchain-project//mlir:LoopLikeInterface",
        "@toolchain-project//mlir:Parser",
        "@toolchain-project//mlir:SideEffectInterfaces",
        "@toolchain-project//mlir:Support",
    ],
)

cc_library(
    name = "machina_ops",
    srcs = [
        "ir/tf_dialect.h",
        "ir/tf_ops.cc",
        "ir/tf_ops.h",
    ],
    textual_hdrs = [
        "ir/tf_all_ops.h.inc",
        "ir/tf_remaining_ops.h",
    ] + ["ir/tf_" + target["name"] + ".h" for target in tf_ops_category_list],
    deps = [
        ":side_effect_analysis_util",
        ":machina_all_ops_inc_gen",
        ":machina_attributes",
        ":machina_op_interfaces",
        ":machina_op_interfaces_inc_gen",
        ":machina_ops_sharded",
        ":machina_remaining_ops",
        ":machina_remaining_ops_inc_gen",
        ":machina_side_effects",
        ":machina_structs",
        ":machina_traits",
        ":machina_types",
        "//machina/compiler/mlir/machina/ir/host_runtime:machina_tfrt_ops",
        "//machina/compiler/mlir/machina/ir/host_runtime:machina_tfrt_ops_inc_gen",
        "//machina/compiler/mlir/machina/transforms:rewrite_util",
        "//machina/compiler/mlir/machina/transforms:machina_canonicalize_inc_gen",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core/common_runtime:inline_function_utils",
        "//machina/core/common_runtime:lower_function_call_inline_policy",
        "@com_google_absl//absl/strings",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:BytecodeOpInterface",
        "@toolchain-project//mlir:CallOpInterfaces",
        "@toolchain-project//mlir:ControlFlowInterfaces",
        "@toolchain-project//mlir:DerivedAttributeOpInterface",
        "@toolchain-project//mlir:Dialect",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:InferTypeOpInterface",
        "@toolchain-project//mlir:InliningUtils",
        "@toolchain-project//mlir:LoopLikeInterface",
        "@toolchain-project//mlir:Parser",
        "@toolchain-project//mlir:SideEffectInterfaces",
        "@toolchain-project//mlir:Support",
    ],
)

cc_library(
    name = "machina_structs",
    srcs = [
        "ir/tf_structs.cc",
    ],
    hdrs = [
        "ir/tf_structs.h",
    ],
    deps = [
        "//machina/core:framework",
        "//machina/core/ir/types:Dialect",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:IR",
    ],
)

cc_library(
    name = "machina_side_effects",
    srcs = [
    ],
    hdrs = [
        "ir/tf_side_effects.h",
    ],
    deps = ["@toolchain-project//mlir:SideEffectInterfaces"],
)

cc_library(
    name = "machina_types",
    hdrs = [
        "ir/tf_dialect.h",
        "ir/tf_types.h",
    ],
    textual_hdrs = [
        "ir/tf_types.def",
    ],
    deps = [
        "//machina/core/ir/types:Dialect",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:Dialect",
        "@toolchain-project//mlir:IR",
    ],
)

cc_library(
    name = "machina",
    srcs = [
        "ir/tf_device.cc",
        "ir/tf_executor.cc",
        "ir/tf_executor.cc.inc",
        "ir/tf_executor.h.inc",
        "ir/tf_saved_model.cc",
    ],
    hdrs = [
        "dialect_registration.h",
        "ir/tf_device.h",
        "ir/tf_dialect.h",
        "ir/tf_executor.h",
        "ir/tf_ops.h",
        "ir/tf_saved_model.h",
        "ir/tf_structs.h",
        "@toolchain-project//mlir:include/mlir/Interfaces/CallInterfaces.h",
    ],
    includes = ["include"],
    visibility = ["//visibility:public"],
    deps = [
        ":machina_all_ops_inc_gen",
        ":machina_attributes",
        ":machina_device_ops_inc_gen",
        ":machina_executor_inc_gen",
        ":machina_op_interfaces",
        ":machina_ops",
        ":machina_side_effects",
        ":machina_structs",
        ":machina_traits",
        ":machina_types",
        ":tf_saved_model_inc_gen",
        "//machina/compiler/mlir/machina/ir/host_runtime:machina_tfrt_ops",
        "//machina/compiler/mlir/machina/ir/host_runtime:machina_tfrt_ops_inc_gen",
        "//machina/compiler/mlir/machina/transforms:machina_canonicalize_inc_gen",
        "//machina/compiler/mlir/machina/transforms:tf_device_pass_inc_gen",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core/ir:Dialect",
        "//machina/core/ir/types:Dialect",
        "//machina/core/platform:logging",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:Analysis",
        "@toolchain-project//mlir:ArithDialect",
        "@toolchain-project//mlir:BytecodeOpInterface",
        "@toolchain-project//mlir:CallOpInterfacesIncGen",
        "@toolchain-project//mlir:ControlFlowDialect",
        "@toolchain-project//mlir:ControlFlowInterfaces",
        "@toolchain-project//mlir:DerivedAttributeOpInterface",
        "@toolchain-project//mlir:Dialect",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:FuncExtensions",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:InferTypeOpInterface",
        "@toolchain-project//mlir:InliningUtils",
        "@toolchain-project//mlir:LoopLikeInterface",
        "@toolchain-project//mlir:MLProgramDialect",
        "@toolchain-project//mlir:Parser",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:SideEffectInterfaces",
        "@toolchain-project//mlir:Support",
        "@toolchain-project//mlir:TransformUtils",
        "@toolchain-project//mlir:Transforms",
    ],
)

tf_cc_test(
    name = "tf_saved_model_test",
    srcs = ["ir/tf_saved_model_test.cc"],
    deps = [
        ":machina",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core/platform:test",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Parser",
        "@toolchain-project//mlir:Support",
    ],
)

cc_library(
    name = "string_util",
    srcs = ["utils/string_util.cc"],
    hdrs = ["utils/string_util.h"],
    deps = [
        "@com_google_absl//absl/strings",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Pass",
    ],
)

cc_library(
    name = "fake_session",
    srcs = ["utils/fake_session.cc"],
    hdrs = ["utils/fake_session.h"],
    deps = [
        "//machina/core:core_cpu_base",
        "//machina/core:framework",
        "//machina/core:framework_internal",
        "//machina/core:protos_all_cc",
        "//machina/core:session_options",
        "//machina/core/common_runtime:threadpool_device",
        "//machina/core/platform:errors",
        "//machina/core/platform:status",
        "//machina/core/platform:threadpool_options",
        "@com_google_absl//absl/strings",
        "@toolchain-project//toolchain:Support",
    ],
)

cc_library(
    name = "session_utils",
    srcs = ["utils/session_utils.cc"],
    hdrs = ["utils/session_utils.h"],
    deps = [
        ":machina",
        ":machina_ops",
        "//machina/compiler/mlir/utils:string_container_utils",
        "//machina/core:core_cpu_base",
        "//machina/core:framework",
        "//machina/core:framework_internal",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Support",
    ],
)

cc_library(
    name = "topological_sort",
    srcs = ["utils/topological_sort.cc"],
    hdrs = ["utils/topological_sort.h"],
    deps = [
        "@com_google_absl//absl/types:span",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Support",
    ],
)

cc_library(
    name = "machina_analysis",
    srcs = [
        "analysis/per_function_aggregate_analysis.h",
        "analysis/resource_alias_analysis.cc",
        "analysis/resource_dataflow.cc",
        "analysis/side_effect_analysis.cc",
    ],
    hdrs = [
        "analysis/resource_alias_analysis.h",
        "analysis/resource_dataflow.h",
        "analysis/side_effect_analysis.h",
        "analysis/tf_dataflow.h",
    ],
    deps = [
        ":machina",
        ":machina_op_interfaces",
        ":machina_side_effects",
        ":machina_traits",
        ":machina_types",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/log",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:Analysis",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:SideEffectInterfaces",
        "@toolchain-project//mlir:Support",
    ],
)

cc_library(
    name = "xla_call_module_attrs",
    srcs = [],
    hdrs = ["utils/xla_call_module_attrs.h"],
    deps = ["@toolchain-project//toolchain:Support"],
)

cc_library(
    name = "stablehlo_custom_call_utils",
    srcs = ["utils/stablehlo_custom_call.cc"],
    hdrs = ["utils/stablehlo_custom_call.h"],
    deps = [
        ":xla_call_module_attrs",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Support",
        "@stablehlo//:stablehlo_ops",
    ],
)

cc_library(
    name = "parse_text_proto",
    srcs = ["utils/parse_text_proto.cc"],
    hdrs = ["utils/parse_text_proto.h"],
    deps = [
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core/platform:casts",
        "@com_google_absl//absl/strings",
        "@com_google_protobuf//:protobuf",
    ],
)

cc_library(
    name = "import_utils",
    srcs = ["utils/import_utils.cc"],
    hdrs = ["utils/import_utils.h"],
    deps = [
        ":error_util",
        ":parse_text_proto",
        "//machina/core:lib",
        "@com_google_absl//absl/strings",
        "@toolchain-project//toolchain:Support",
    ],
)

cc_library(
    name = "export_utils",
    srcs = [
        "utils/export_utils.cc",
    ],
    hdrs = [
        "utils/export_utils.h",
    ],
    deps = [
        ":attribute_utils",
        ":convert_tensor",
        ":convert_type",
        ":location_utils",
        ":mangling_util",
        ":machina",
        ":machina_attributes",
        ":machina_types",
        "//machina/compiler/mlir/machina:xla_sharding_util",
        "//machina/core:framework",
        "//machina/core:graph",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core/platform:protobuf",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Support",
        "@local_xla//xla:status_macros",
        "@local_xla//xla:xla_data_proto_cc",
        "@local_xla//xla/hlo/builder:sharding_builder",
        "@local_xla//xla/hlo/ir:hlo",
        "@local_xla//xla/hlo/parser:hlo_parser",
    ],
)

cc_library(
    name = "location_utils",
    srcs = ["utils/location_utils.cc"],
    hdrs = ["utils/location_utils.h"],
    deps = [
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Support",
    ],
)

cc_library(
    name = "tpu_cluster_util",
    srcs = ["utils/tpu_cluster_util.cc"],
    hdrs = ["utils/tpu_cluster_util.h"],
    deps = [
        ":device_util",
        ":machina",
        ":machina_structs",
        ":tpu_rewrite_device_util",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:Analysis",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Support",
    ],
)

cc_library(
    name = "translate_utils",
    srcs = [
        "utils/translate_utils.cc",
    ],
    hdrs = [
        "utils/translate_utils.h",
    ],
    deps = [
        ":convert_type",
        ":export_utils",
        ":machina",
        ":machina_types",
        "//machina/compiler/mlir/utils:string_container_utils",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:DerivedAttributeOpInterface",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Support",
    ],
)

cc_library(
    name = "convert_attr",
    srcs = ["utils/convert_attr.cc"],
    hdrs = ["utils/convert_attr.h"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        ":convert_tensor",
        ":convert_type",
        ":machina_attributes",
        "//machina/core:protos_all_cc",
        "//machina/core/platform:errors",
        "@toolchain-project//mlir:IR",
        "@local_tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "convert_type",
    srcs = [
        "utils/convert_type.cc",
    ],
    hdrs = [
        "utils/convert_type.h",
    ],
    textual_hdrs = [
        "ir/tf_types.def",
    ],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        ":dynamic_shape_utils",
        ":machina_types",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "@com_google_absl//absl/strings",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Support",
    ],
)

tf_cc_test(
    name = "convert_type_test",
    size = "small",
    srcs = ["utils/convert_type_test.cc"],
    deps = [
        ":convert_type",
        "//machina/core:lib",
        "//machina/core:test",
        "//machina/core:test_main",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:IR",
    ],
)

cc_library(
    name = "convert_tensor",
    srcs = ["utils/convert_tensor.cc"],
    hdrs = ["utils/convert_tensor.h"],
    deps = [
        ":convert_type",
        ":dynamic_shape_utils",
        ":mangling_util",
        ":machina_attributes",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:cord",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Support",
        "@local_tsl//tsl/platform:ml_dtypes",
    ],
)

tf_cc_test(
    name = "convert_tensor_test",
    size = "small",
    srcs = ["utils/convert_tensor_test.cc"],
    deps = [
        ":convert_tensor",
        ":dynamic_shape_utils",
        ":machina",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core:testlib",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Support",
        "@local_tsl//tsl/platform:ml_dtypes",
        "@local_xla//xla/hlo/testlib:test",
    ],
)

cc_library(
    name = "mangling_util",
    srcs = ["utils/mangling_util.cc"],
    hdrs = ["utils/mangling_util.h"],
    deps = [
        ":parse_text_proto",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core/ir/importexport:mangling",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "error_util",
    srcs = ["utils/error_util.cc"],
    hdrs = ["utils/error_util.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//machina/core/platform:errors",
        "//machina/core/platform:status",
        "//machina/core/util:managed_stack_trace",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:string_view",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Support",
        "@local_xla//xla/mlir/utils:error_util",
    ],
)

cc_library(
    name = "tf_dialect_lib",
    deps = [
        "//machina/compiler/mlir/machina/transforms:tf_dialect_passes",
        "@toolchain-project//mlir:AllPassesAndDialects",
    ],
)

cc_library(
    name = "eval_util",
    srcs = ["utils/eval_util.cc"],
    hdrs = ["utils/eval_util.h"],
    deps = [
        ":convert_tensor",
        ":translate_utils",
        "//machina/c/eager:c_api",
        "//machina/c/eager:c_api_internal",
        "//machina/core:framework",
        "//machina/core:framework_internal",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Support",
    ],
)

tf_cc_test(
    name = "error_util_test",
    srcs = ["utils/error_util_test.cc"],
    deps = [
        ":error_util",
        "//machina/core:lib",
        "//machina/core:test",
        "//machina/core:test_main",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:IR",
        "@local_xla//xla/hlo/testlib:test",
        "@local_xla//xla/mlir/utils:error_util",
    ],
)

cc_library(
    name = "deserialize_mlir_module_utils",
    srcs = ["utils/deserialize_mlir_module_utils.cc"],
    hdrs = ["utils/deserialize_mlir_module_utils.h"],
    deps = [
        ":error_util",
        "//machina/core/platform:status",
        "@com_google_absl//absl/status",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Parser",
        "@local_xla//xla:status_macros",
    ],
)

cc_library(
    name = "serialize_mlir_module_utils",
    srcs = ["utils/serialize_mlir_module_utils.cc"],
    hdrs = ["utils/serialize_mlir_module_utils.h"],
    deps = [
        "//machina/compiler/jit:flags",
        "//machina/compiler/jit:flags_headers",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:IR",
    ],
)

tf_cc_test(
    name = "serialize_mlir_module_utils_test",
    srcs = ["utils/serialize_mlir_module_utils_test.cc"],
    deps = [
        ":serialize_mlir_module_utils",
        "//machina/compiler/jit:flags",
        "//machina/core:test",
        "//machina/core:test_main",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@toolchain-project//mlir:IR",
    ],
)

cc_library(
    name = "tf_xla_mlir_translate",
    testonly = True,  # Ensure alwayslink does not leak in the codebase.
    srcs = ["utils/tf_xla_mlir_translate.cc"],
    deps = [
        ":deserialize_mlir_module_utils",
        ":mlir_roundtrip_flags",
        ":serialize_mlir_module_utils",
        ":machina",
        "//machina/compiler/mlir/machina/translate/tools:parsers",
        "//machina/compiler/mlir/tf2xla:compile_mlir_util",
        "//machina/compiler/mlir/tools:translate_cl_options",
        "//machina/compiler/mlir/utils:string_container_utils",
        "//machina/compiler/tf2xla:layout_util",
        "//machina/compiler/tf2xla:xla_argument",
        "//machina/compiler/tf2xla:xla_helpers",
        "//machina/core:framework",
        "//machina/core:protos_all_cc",
        "//machina/core/platform:errors",
        "//machina/core/platform:status",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:ArithDialect",
        "@toolchain-project//mlir:AsmParser",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:FuncExtensions",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:QuantOps",
        "@toolchain-project//mlir:Support",
        "@toolchain-project//mlir:TranslateLib",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:statusor",
        "@local_xla//xla:shape_util",
        "@local_xla//xla/hlo/builder:xla_builder",
        "@local_xla//xla/hlo/builder:xla_computation",
        "@local_xla//xla/hlo/ir:hlo",
        "@local_xla//xla/hlo/translate/mhlo_to_hlo:type_to_shape",
        "@local_xla//xla/service:hlo_module_config",
        "@local_xla//xla/service:hlo_proto_cc",
        "@stablehlo//:stablehlo_ops",
    ],
    alwayslink = 1,
)

cc_library(
    name = "mlir_passthrough_op",
    srcs = ["ops/mlir_passthrough_op.cc"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "//machina/core:framework",
    ],
    alwayslink = 1,
)

cc_library(
    name = "mlir_local_var_op",
    srcs = ["ops/mlir_local_var_op.cc"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "//machina/core:framework",
    ],
    alwayslink = 1,
)

tf_gen_op_wrapper_py(
    name = "gen_mlir_passthrough_op_py",
    out = "gen_mlir_passthrough_op.py",
    compatible_with = [],
    extra_py_deps = [
        "//machina/python:pywrap_tfe",
        "//machina/python/util:dispatch",
        "//machina/python/util:deprecation",
        "//machina/python/util:tf_export",
    ],
    py_lib_rule = py_strict_library,
    deps = [":mlir_passthrough_op"],
)

cc_library(
    name = "parallel_execute_util",
    srcs = ["utils/parallel_execute_util.cc"],
    hdrs = ["utils/parallel_execute_util.h"],
    deps = [
        ":machina",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Support",
    ],
)

cc_library(
    name = "cluster_util",
    srcs = ["utils/cluster_util.cc"],
    hdrs = ["utils/cluster_util.h"],
    deps = [
        ":machina_analysis",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Support",
        "@toolchain-project//mlir:TransformUtils",
    ],
)

tf_cc_test(
    name = "cluster_util_test",
    size = "small",
    srcs = ["utils/cluster_util_test.cc"],
    deps = [
        ":cluster_util",
        ":deserialize_mlir_module_utils",
        ":machina",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core/platform:errors",
        "@com_google_googletest//:gtest",
        "@toolchain-project//mlir:FuncDialect",
        "@local_tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "tpu_rewrite_device_util",
    srcs = ["utils/tpu_rewrite_device_util.cc"],
    hdrs = ["utils/tpu_rewrite_device_util.h"],
    deps = [
        ":device_util",
        ":machina",
        ":machina_types",
        "//machina/compiler/jit:flags_headers",
        "//machina/compiler/mlir/utils:string_container_utils",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core/protobuf/tpu:topology_proto_cc",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Support",
        "@local_xla//xla:array4d",
        "@local_xla//xla:xla_data_proto_cc",
        "@local_xla//xla/service:computation_placer",
    ],
)

tf_cc_test(
    name = "tpu_rewrite_device_util_test",
    size = "small",
    srcs = ["utils/tpu_rewrite_device_util_test.cc"],
    deps = [
        ":deserialize_mlir_module_utils",
        ":device_util",
        ":machina",
        ":tpu_rewrite_device_util",
        "//machina/compiler/jit:flags",
        "//machina/core:framework",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core/platform:status_matchers",
        "//machina/core/protobuf/tpu:topology_proto_cc",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest_main",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Support",
        "@local_tsl//tsl/platform:statusor",
        "@local_xla//xla:xla_data_proto_cc",
    ],
)

cc_library(
    name = "device_util",
    srcs = ["utils/device_util.cc"],
    hdrs = ["utils/device_util.h"],
    deps = [
        ":machina",
        "//machina/core:core_cpu_lib",
        "//machina/core:framework",
        "@com_google_absl//absl/strings",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Support",
    ],
)

tf_cc_test(
    name = "device_util_test",
    size = "small",
    srcs = ["utils/device_util_test.cc"],
    deps = [
        ":device_util",
        "//machina/core:core_cpu_lib",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core/ir/types:Dialect",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Support",
    ],
)

cc_library(
    name = "dump_mlir_util",
    srcs = ["utils/dump_mlir_util.cc"],
    hdrs = ["utils/dump_mlir_util.h"],
    deps = [
        "//machina/core:lib",
        "//machina/core/platform:crash_analysis",
        "//machina/core/platform:logging",
        "@com_google_absl//absl/strings",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Pass",
        "@local_xla//xla/tsl/lib/io:buffered_file",
    ],
)

tf_cc_test(
    name = "dump_mlir_util_test",
    size = "small",
    srcs = ["utils/dump_mlir_util_test.cc"],
    deps = [
        ":dump_mlir_util",
        ":machina",
        "//machina/compiler/mlir/machina/transforms:bridge",
        "//machina/compiler/mlir/machina/transforms:machina_passes",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core/platform:test",
        "@com_google_googletest//:gtest",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:AllPassesAndDialects",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:MlirOptLib",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:RegisterAllDialects",
        "@toolchain-project//mlir:RegisterAllPasses",
        "@toolchain-project//mlir:Support",
    ],
)

cc_library(
    name = "dump_graph",
    srcs = ["utils/dump_graph.cc"],
    hdrs = ["utils/dump_graph.h"],
    deps = [
        ":error_util",
        ":machina",
        "//machina/core:framework",
        "//machina/core:graph",
        "//machina/core:lib",
        "//machina/core/ir/importexport:graphdef_import",
        "//machina/core/platform:logging",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:IR",
    ],
)

tf_cc_test(
    name = "dump_graph_test",
    size = "small",
    srcs = ["utils/dump_graph_test.cc"],
    tags = [
        "no_windows",  # b/208469759
    ],
    deps = [
        ":dump_graph",
        "//machina/core:framework",
        "//machina/core:graph",
        "//machina/core:lib",
        "//machina/core:ops",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core/platform:test",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:IR",
    ],
)

tf_cc_test(
    name = "bridge_logger_test",
    size = "small",
    srcs = ["utils/bridge_logger_test.cc"],
    deps = [
        ":bridge_logger",
        ":deserialize_mlir_module_utils",
        ":machina",
        "//machina/compiler/mlir/machina/transforms:machina_passes",
        "//machina/core:lib",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core/platform:test",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:Transforms",
    ],
)

tf_cc_test(
    name = "data_dumper_logger_config_test",
    size = "small",
    srcs = ["utils/data_dumper_logger_config_test.cc"],
    deps = [
        ":bridge_logger",
        ":deserialize_mlir_module_utils",
        ":machina",
        "//machina/compiler/mlir/machina/transforms:machina_passes",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core/platform:test",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Transforms",
    ],
)

cc_library(
    name = "bridge_logger",
    srcs = [
        "utils/bridge_logger.cc",
        "utils/data_dumper_logger_config.cc",
    ],
    hdrs = [
        "utils/bridge_logger.h",
        "utils/data_dumper_logger_config.h",
    ],
    deps = [
        ":dump_mlir_util",
        "//machina/core:lib",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:Support",
    ],
)

cc_library(
    name = "call_graph_util",
    srcs = [
        "utils/call_graph_util.cc",
    ],
    hdrs = [
        "utils/call_graph_util.h",
    ],
    deps = [
        ":machina",
        "@com_google_absl//absl/strings",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Support",
    ],
)

tf_cc_test(
    name = "call_graph_util_test",
    size = "small",
    srcs = ["utils/call_graph_util_test.cc"],
    deps = [
        ":attribute_utils",
        ":call_graph_util",
        ":machina",
        "//machina/core:test",
        "//machina/core:test_main",
        "@com_google_googletest//:gtest_main",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Parser",
    ],
)

cc_library(
    name = "xla_sharding_util",
    srcs = [
        "utils/xla_sharding_util.cc",
    ],
    hdrs = [
        "utils/xla_sharding_util.h",
    ],
    deps = [
        ":machina",
        "//machina/core:framework",
        "//machina/core/protobuf/tpu:compile_metadata_proto_cc",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Support",
        "@local_xla//xla:xla_data_proto_cc",
        "@local_xla//xla/hlo/ir:hlo",
        "@local_xla//xla/hlo/ir:tile_assignment",
        "@local_xla//xla/hlo/parser:hlo_parser",
        "@local_xla//xla/tsl/lib/math:math_util",
    ],
)

tf_cc_test(
    name = "utils/xla_sharding_util_test",
    srcs = ["utils/xla_sharding_util_test.cc"],
    deps = [
        ":deserialize_mlir_module_utils",
        ":machina",
        ":xla_sharding_util",
        "//machina/core:test",
        "//machina/core/protobuf/tpu:topology_proto_cc",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest_main",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Support",
        "@local_tsl//tsl/platform:statusor",
        "@local_xla//xla:xla_data_proto_cc",
    ],
)

cc_library(
    name = "attribute_utils",
    srcs = ["utils/attribute_utils.cc"],
    hdrs = ["utils/attribute_utils.h"],
    deps = [
        "//machina/compiler/tf2xla:tf2xla_defs",
        "@com_google_absl//absl/strings",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Support",
    ],
)

cc_library(
    name = "shape_inference_utils",
    srcs = ["utils/shape_inference_utils.cc"],
    hdrs = ["utils/shape_inference_utils.h"],
    deps = [
        ":machina_types",
        ":translate_utils",
        "//machina/core/ir:shape_inference_utils",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:InferTypeOpInterface",
        "@toolchain-project//mlir:Support",
    ],
)

cc_library(
    name = "dynamic_shape_utils",
    srcs = ["utils/dynamic_shape_utils.cc"],
    hdrs = ["utils/dynamic_shape_utils.h"],
    deps = [
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:IR",
    ],
)

cc_library(
    name = "verification_utils",
    srcs = ["utils/verification_utils.cc"],
    hdrs = ["utils/verification_utils.h"],
    deps = [
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Support",
    ],
)

cc_library(
    name = "verify_suitable_for_graph_export",
    srcs = ["utils/verify_suitable_for_graph_export.cc"],
    hdrs = ["utils/verify_suitable_for_graph_export.h"],
    deps = [
        ":machina",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Support",
    ],
)

cc_library(
    name = "resource_value_typed_analyzer",
    srcs = ["analysis/resource_value_typed_analyzer.cc"],
    hdrs = ["analysis/resource_value_typed_analyzer.h"],
    deps = [
        ":machina",
        ":machina_types",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Support",
        "@toolchain-project//mlir:TransformUtils",
    ],
)

cc_library(
    name = "tf_arith_ops_folder",
    srcs = ["ir/tf_arith_ops_folder.cc"],
    hdrs = ["ir/tf_arith_ops_folder.h"],
    deps = [
        ":machina_types",
        "//machina/core/ir/types:Dialect",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:Dialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Support",
    ],
)

cc_library(
    name = "tf_ops_canonicalization_helper",
    hdrs = ["ir/tf_ops_canonicalization_helper.h"],
    deps = [
        ":attribute_utils",
        "@toolchain-project//mlir:IR",
    ],
)

cc_library(
    name = "tf_ops_device_helper",
    srcs = ["ir/tf_ops_device_helper.cc"],
    hdrs = ["ir/tf_ops_device_helper.h"],
    deps = [
        ":machina_structs",
        "//machina/core:framework",
        "//machina/core:framework_internal",
    ],
)

cc_library(
    name = "tf_ops_layout_helper",
    srcs = ["ir/tf_ops_layout_helper.cc"],
    hdrs = ["ir/tf_ops_layout_helper.h"],
    deps = [
        ":machina_op_interfaces",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Support",
    ],
)

cc_library(
    name = "tf_ops_tensor_helper",
    srcs = ["ir/tf_ops_tensor_helper.cc"],
    hdrs = ["ir/tf_ops_tensor_helper.h"],
    deps = [
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:Dialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Support",
    ],
)

cc_library(
    name = "tpu_embedding_ops_registry",
    srcs = [
        "ir/tpu_embedding_ops_registry.cc",
    ],
    hdrs = [
        "ir/tpu_embedding_ops_registry.h",
    ],
    deps = [
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Support",
    ],
)

cc_library(
    name = "mlprogram_util",
    srcs = ["utils/mlprogram_util.cc"],
    hdrs = ["utils/mlprogram_util.h"],
    deps = [
        "//machina/compiler/mlir/machina/transforms:mlprogram",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:Support",
    ],
)

cc_library(
    name = "visitor",
    srcs = ["utils/visitor.cc"],
    hdrs = ["utils/visitor.h"],
    deps = [
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Support",
    ],
)

cc_library(
    name = "xla_rewrite_util",
    srcs = ["utils/xla_rewrite_util.cc"],
    hdrs = ["utils/xla_rewrite_util.h"],
    deps = [
        ":device_util",
        ":machina",
        ":machina_types",
        "//machina/compiler/mlir/utils:string_container_utils",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core/protobuf/tpu:topology_proto_cc",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Support",
        "@local_xla//xla:array4d",
        "@local_xla//xla:xla_data_proto_cc",
        "@local_xla//xla/service:computation_placer",
    ],
)

tf_cc_test(
    name = "xla_rewrite_util_test",
    size = "small",
    srcs = ["utils/xla_rewrite_util_test.cc"],
    deps = [
        ":deserialize_mlir_module_utils",
        ":device_util",
        ":machina",
        ":tpu_rewrite_device_util",
        ":xla_rewrite_util",
        "//machina/compiler/jit:flags",
        "//machina/core:framework",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core/protobuf/tpu:topology_proto_cc",
        "@com_google_absl//absl/status:statusor",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Support",
        "@local_tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "side_effect_analysis_util",
    srcs = [
        "utils/side_effect_analysis_util.cc",
    ],
    hdrs = [
        "utils/side_effect_analysis_util.h",
    ],
    deps = [
        "machina_side_effects",
        "machina_types",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:SideEffectInterfaces",
        "@toolchain-project//mlir:Support",
    ],
)

build_test(
    name = "machina_build_test",
    targets = [
        ":machina",
    ],
)
# copybara:uncomment_begin(google-only)
#
# # Generate new_tf_generated_ops.td. Without --update-existing, this file can
# # replace tf_generated_ops.td. However, right now, we only use it to determine
# # whether the latter is stale.
# dialectgen(
#     name = "tf-generated-ops",
#     output = "new_tf_generated_ops.td",
# )
#
# sh_test(
#     name = "tf_generated_ops_not_stale",
#     srcs = ["ir/tf_generated_ops_not_stale.sh"],
#     args = [
#         "$(location //machina/compiler/mlir/machina:ir/tf_generated_ops.td)",
#         "$(location :new_tf_generated_ops.td)",
#     ],
#     data = [
#         ":new_tf_generated_ops.td",
#         "//machina/compiler/mlir/machina:ir/tf_generated_ops.td",
#     ],
# )
#
# copybara:uncomment_end(google-only)

# Required as we created the transforms subpackage and need to update
# these BUILD targets in a follow up.
aliased_targets = [
    "import_model",
    "mlir_roundtrip_flags",
    "mlir_import_options",
    "translate_lib",
]

[
    alias(
        name = target,
        actual = "//machina/compiler/mlir/machina/translate:%s" % target,
        visibility = ["//visibility:public"],
    )
    for target in aliased_targets
]

alias(
    name = "file_tf_mlir_translate",
    actual = "//machina/compiler/mlir/machina/translate/tools:file_tf_mlir_translate",
    visibility = ["//visibility:public"],
)

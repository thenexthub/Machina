load("@toolchain-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//machina:machina.bzl", "tf_cc_test")
load("//machina:machina.default.bzl", "get_compatible_with_portable")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_compatible_with = get_compatible_with_portable(),
    default_visibility = [
        "//machina/compiler/mlir/machina:__subpackages__",
        "//machina/core:__subpackages__",
        "//machina/tools/tfg_graph_transforms:__subpackages__",
    ],
    licenses = ["notice"],  # Apache 2.0
)

td_library(
    name = "InterfacesTdFiles",
    srcs = ["interfaces.td"],
    deps = [
        "@toolchain-project//mlir:OpBaseTdFiles",
    ],
)

gentbl_cc_library(
    name = "InterfacesIncGen",
    tbl_outs = {
        "interfaces.h.inc": ["-gen-op-interface-decls"],
        "interfaces.cc.inc": ["-gen-op-interface-defs"],
    },
    tblgen = "@toolchain-project//mlir:mlir-tblgen",
    td_file = "interfaces.td",
    deps = [
        ":InterfacesTdFiles",
    ],
)

# ODS (https://mlir.toolchain.org/docs/OpDefinitions/) generation for op and dialect files to include.
td_library(
    name = "DialectTdFiles",
    srcs = [
        "dialect.td",
        "ops.td",
    ],
    deps = [
        "@toolchain-project//mlir:CallInterfacesTdFiles",
        "@toolchain-project//mlir:ControlFlowInterfacesTdFiles",
        "@toolchain-project//mlir:FunctionInterfacesTdFiles",
        "@toolchain-project//mlir:InferTypeOpInterfaceTdFiles",
        "@toolchain-project//mlir:OpBaseTdFiles",
        "@toolchain-project//mlir:SideEffectInterfacesTdFiles",
    ],
)

gentbl_cc_library(
    name = "DialectIncGen",
    tbl_outs = {
        "ops.h.inc": [
            "-gen-op-decls",
            "-dialect",
            "tfg",
        ],
        "ops.cc.inc": [
            "-gen-op-defs",
            "-dialect",
            "tfg",
        ],
        "dialect.h.inc": [
            "-gen-dialect-decls",
            "-dialect",
            "tfg",
        ],
        "dialect.cc.inc": [
            "-gen-dialect-defs",
            "-dialect",
            "tfg",
        ],
        "attributes.h.inc": [
            "-gen-attrdef-decls",
            "-attrdefs-dialect",
            "tfg",
        ],
        "attributes.cc.inc": [
            "-gen-attrdef-defs",
            "-attrdefs-dialect",
            "tfg",
        ],
    },
    tblgen = "@toolchain-project//mlir:mlir-tblgen",
    td_file = "ops.td",
    deps = [
        ":DialectTdFiles",
        ":InterfacesTdFiles",
        "//machina/core/ir/types:DialectTdFiles",
    ],
)

cc_library(
    name = "Dialect",
    srcs = [
        "interfaces.cc",
        "ops.cc",
        "tf_op_names.cc",
        "tf_op_names.inc",
        "tf_op_wrapper.cc",
        "utility.cc",
    ],
    hdrs = [
        "dialect.h",
        "interfaces.h",
        "ops.h",
        "tf_op_wrapper.h",
        "utility.h",
    ],
    deps = [
        ":DialectIncGen",
        ":InterfacesIncGen",
        "//machina/core/ir/types:Dialect",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:CallOpInterfaces",
        "@toolchain-project//mlir:ControlFlowInterfaces",
        "@toolchain-project//mlir:FunctionInterfaces",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:InferTypeOpInterface",
        "@toolchain-project//mlir:QuantOps",
        "@toolchain-project//mlir:SideEffectInterfaces",
        "@toolchain-project//mlir:Support",
    ],
)

cc_library(
    name = "tf_op_registry",
    srcs = ["tf_op_registry.cc"],
    hdrs = ["tf_op_registry.h"],
    deps = [
        ":Dialect",
        "//machina/core:framework",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Support",
    ],
)

cc_library(
    name = "shape_inference_utils",
    srcs = ["utils/shape_inference_utils.cc"],
    hdrs = ["utils/shape_inference_utils.h"],
    deps = [
        ":Dialect",
        "//machina/core:framework",
        "//machina/core:protos_all_cc",
        "//machina/core/ir/importexport:convert_tensor",
        "//machina/core/ir/importexport:convert_types",
        "//machina/core/ir/importexport:graphdef_export",
        "//machina/core/ir/types:Dialect",
        "//machina/core/platform:logging",
        "//machina/core/platform:status",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:DerivedAttributeOpInterface",
        "@toolchain-project//mlir:FunctionInterfaces",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:InferTypeOpInterface",
        "@toolchain-project//mlir:Support",
    ],
)

tf_cc_test(
    name = "utility_test",
    size = "small",
    srcs = ["utility_test.cc"],
    deps = [
        ":Dialect",
        "//machina/core:test",
        "//machina/core:test_main",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Parser",
        "@toolchain-project//mlir:Support",
    ],
)

tf_cc_test(
    name = "tf_op_wrapper_test",
    size = "small",
    srcs = ["tf_op_wrapper_test.cc"],
    deps = [
        ":Dialect",
        "//machina/core:test",
        "//machina/core:test_main",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Parser",
        "@toolchain-project//mlir:Support",
    ],
)

tf_cc_test(
    name = "shape_inference_utils_test",
    size = "small",
    srcs = ["utils/shape_inference_utils_test.cc"],
    deps = [
        ":Dialect",
        ":shape_inference_utils",
        "//machina/core:framework",
        "//machina/core:ops",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core/platform:errors",
        "@com_google_absl//absl/status",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:InferTypeOpInterface",
        "@toolchain-project//mlir:Parser",
        "@toolchain-project//mlir:Support",
    ],
)

tf_cc_test(
    name = "ops_test",
    size = "small",
    srcs = ["ops_test.cc"],
    deps = [
        ":Dialect",
        "//machina/core:test",
        "//machina/core:test_main",
        "@toolchain-project//mlir:ControlFlowInterfaces",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Parser",
    ],
)

tf_cc_test(
    name = "interfaces_test",
    size = "small",
    srcs = ["interfaces_test.cc"],
    deps = [
        ":Dialect",
        "//machina/core:test",
        "//machina/core:test_main",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:IR",
    ],
)

tf_cc_test(
    name = "tf_op_registry_test",
    size = "small",
    srcs = ["tf_op_registry_test.cc"],
    deps = [
        ":Dialect",
        ":tf_op_registry",
        "//machina/core:ops",
        "//machina/core:test",
        "//machina/core:test_main",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Parser",
        "@toolchain-project//mlir:Support",
    ],
)

load(
    "@toolchain-project//mlir:tblgen.bzl",
    "gentbl_cc_library",
    "td_library",
)
load("//machina:strict.default.bzl", "py_strict_library")
load(
    "//machina:machina.bzl",
    "tf_cc_binary",
    "tf_cc_test",
)
load("//machina:machina.default.bzl", "get_compatible_with_portable", "tf_py_strict_test", "tf_python_pybind_extension")
load("//machina/compiler/mlir:glob_lit_test.bzl", "glob_lit_tests")
load("//machina/compiler/mlir/tfr:build_defs.bzl", "gen_op_libraries")
load("//machina/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = [
        ":friends",
    ],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    packages = [
        "//machina/c/...",
        "//machina/compiler/...",
        # Allow visibility from the mlir language server.
        "//learning/brain/mlir/mlir_lsp_server/...",
    ],
)

td_library(
    name = "tfr_ops_td_files",
    srcs = [
        "ir/tfr_ops.td",
    ],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//machina/compiler/mlir/machina:machina_ops_td_files",
        "@toolchain-project//mlir:CallInterfacesTdFiles",
        "@toolchain-project//mlir:ControlFlowInterfacesTdFiles",
        "@toolchain-project//mlir:FunctionInterfacesTdFiles",
        "@toolchain-project//mlir:OpBaseTdFiles",
        "@toolchain-project//mlir:QuantizationOpsTdFiles",
        "@toolchain-project//mlir:ShapeOpsTdFiles",
        "@toolchain-project//mlir:SideEffectInterfacesTdFiles",
    ],
)

gentbl_cc_library(
    name = "tfr_ops_inc_gen",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = {
        "ir/tfr_ops.h.inc": ["-gen-op-decls"],
        "ir/tfr_ops.cc.inc": ["-gen-op-defs"],
    },
    tblgen = "@toolchain-project//mlir:mlir-tblgen",
    td_file = "ir/tfr_ops.td",
    deps = [
        ":tfr_ops_td_files",
    ],
)

gentbl_cc_library(
    name = "tfr_decompose_inc_gen",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = {"passes/generated_decompose.inc": ["-gen-rewriters"]},
    tblgen = "@toolchain-project//mlir:mlir-tblgen",
    td_file = "passes/decompose_patterns.td",
    deps = [
        ":tfr_ops_td_files",
        "@toolchain-project//mlir:ArithOpsTdFiles",
        "@toolchain-project//mlir:FuncTdFiles",
    ],
)

cc_library(
    name = "tfr",
    srcs = [
        "ir/tfr_ops.cc",
        "ir/tfr_ops.cc.inc",
    ],
    hdrs = [
        "ir/tfr_ops.h",
        "ir/tfr_ops.h.inc",
        "ir/tfr_types.h",
    ],
    deps = [
        ":tfr_ops_inc_gen",
        "//machina/compiler/mlir/machina",
        "//machina/compiler/mlir/machina:machina_attributes",
        "//machina/compiler/mlir/machina:machina_ops",
        "//machina/compiler/mlir/machina:machina_types",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:ArithDialect",
        "@toolchain-project//mlir:ControlFlowInterfaces",
        "@toolchain-project//mlir:Dialect",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:FunctionInterfaces",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:InferTypeOpInterface",
        "@toolchain-project//mlir:InliningUtils",
        "@toolchain-project//mlir:QuantOps",
        "@toolchain-project//mlir:ShapeDialect",
        "@toolchain-project//mlir:SideEffectInterfaces",
        "@toolchain-project//mlir:Support",
        "@toolchain-project//mlir:TransformUtils",
    ],
)

cc_library(
    name = "utils",
    srcs = [
        "utils/utils.cc",
    ],
    hdrs = [
        "utils/utils.h",
    ],
    deps = [
        ":tfr",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Support",
    ],
)

cc_library(
    name = "passes",
    srcs = [
        "passes/canonicalize.cc",
        "passes/decompose.cc",
        "passes/generated_decompose.inc",
        "passes/raise_to_tf.cc",
        "passes/rewrite_quantized_io.cc",
    ],
    hdrs = [
        "passes/passes.h",
    ],
    deps = [
        ":tfr",
        ":utils",
        "//machina/compiler/mlir/machina",
        "//machina/core:lib",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:AffineUtils",
        "@toolchain-project//mlir:ArithDialect",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:InliningUtils",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:QuantOps",
        "@toolchain-project//mlir:SCFDialect",
        "@toolchain-project//mlir:SCFToControlFlow",
        "@toolchain-project//mlir:Support",
        "@toolchain-project//mlir:TransformUtils",
    ],
    alwayslink = 1,
)

tf_cc_binary(
    name = "tfr-opt",
    srcs = ["passes/tfr_opt.cc"],
    deps = [
        ":passes",
        ":tfr",
        "//machina/compiler/mlir:init_mlir",
        "//machina/compiler/mlir:passes",
        "//machina/compiler/mlir/quantization/common/ir:QuantOps",
        "//machina/compiler/mlir/machina",
        "//machina/compiler/mlir/machina/transforms:machina_passes",
        "@toolchain-project//mlir:AllPassesAndDialects",
        "@toolchain-project//mlir:ArithDialect",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:FuncExtensions",
        "@toolchain-project//mlir:MlirOptLib",
        "@toolchain-project//mlir:QuantOps",
        "@toolchain-project//mlir:RegisterAllPasses",
        "@toolchain-project//mlir:SCFDialect",
        "@toolchain-project//mlir:ShapeDialect",
    ],
)

glob_lit_tests(
    name = "all_tests",
    data = [":test_utilities"],
    driver = "//machina/compiler/mlir:run_lit.sh",
    test_file_exts = ["mlir"],
)

# Bundle together all of the test utilities that are used by tests.
filegroup(
    name = "test_utilities",
    testonly = True,
    data = [
        ":tfr-opt",
        "@toolchain-project//toolchain:FileCheck",
        "@toolchain-project//toolchain:not",
        "@toolchain-project//mlir:run_lit.sh",
    ],
)

cc_library(
    name = "tfr_decompose_ctx",
    srcs = ["integration/tfr_decompose_ctx.cc"],
    hdrs = ["integration/tfr_decompose_ctx.h"],
    deps = [
        ":passes",
        ":tfr",
        "//machina/compiler/mlir/machina",
        "//machina/compiler/mlir/machina:convert_attr",
        "//machina/compiler/mlir/machina:convert_type",
        "//machina/compiler/mlir/machina:mlir_roundtrip_flags",
        "//machina/compiler/mlir/machina/transforms:machina_passes",
        "//machina/compiler/mlir/tf2xla/api/v2:tf_executor_to_graph",
        "//machina/core:framework",
        "//machina/core:framework_types_hdr",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core:protos_all_cc",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:ArithDialect",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:FuncExtensions",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Parser",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:SCFDialect",
        "@toolchain-project//mlir:ShapeDialect",
    ],
)

tf_cc_test(
    name = "tfr_decompose_ctx_test",
    srcs = ["integration/tfr_decompose_ctx_test.cc"],
    deps = [
        ":tfr_decompose_ctx",
        "//machina/core:framework",
        "//machina/core:ops",
        "//machina/core:protos_all_cc",
        "//machina/core:test",
        "//machina/core:test_main",
        "@com_google_absl//absl/types:span",
        "@toolchain-project//mlir:IR",
        "@local_xla//xla/hlo/testlib:test",
    ],
)

tf_python_pybind_extension(
    name = "tfr_wrapper",
    srcs = ["python/tfr_wrapper.cc"],
    enable_stub_generation = True,
    pytype_srcs = [
        "tfr_wrapper.pyi",
    ],
    starlark_only = True,
    visibility = [
        "//machina/python:__pkg__",
    ],
    deps = [
        ":tfr",
        "//machina/compiler/mlir/machina",
        "//machina/python/lib/core:pybind11_lib",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:ArithDialect",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Parser",
        "@toolchain-project//mlir:SCFDialect",
        "@toolchain-project//mlir:ShapeDialect",
        "@pybind11",
    ],
)

py_strict_library(
    name = "composite",
    srcs = ["python/composite.py"],
)

py_strict_library(
    name = "tfr_gen",
    srcs = ["python/tfr_gen.py"],
    deps = [
        ":tfr_wrapper",
        "//machina:machina_py",  # buildcleaner: keep
        "//machina/core:protos_all_py",
        "//machina/python/autograph/converters:control_flow",
        "//machina/python/autograph/converters:return_statements",
        "//machina/python/autograph/impl:api",
        "//machina/python/autograph/pyct:anno",
        "//machina/python/autograph/pyct:cfg",
        "//machina/python/autograph/pyct:qual_names",
        "//machina/python/autograph/pyct:transformer",
        "//machina/python/autograph/pyct:transpiler",
        "//machina/python/autograph/pyct/static_analysis:activity",
        "//machina/python/autograph/pyct/static_analysis:reaching_definitions",
        "//machina/python/autograph/pyct/static_analysis:reaching_fndefs",
        "//machina/python/autograph/pyct/static_analysis:type_inference",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:load_library",
        "//machina/python/framework:op_def_registry",
        "//machina/python/platform:tf_logging",
        "//machina/python/util:tf_inspect",
        "@pypi_gast//:pkg",
    ],
)

tf_py_strict_test(
    name = "tfr_gen_test",
    size = "medium",
    srcs = ["python/tfr_gen_test.py"],
    deps = [
        ":composite",
        ":tfr_gen",
        "//machina/compiler/mlir/python/mlir_wrapper:filecheck_wrapper",
        "//machina/compiler/mlir/tfr/resources:test_ops",
        "//machina/python/framework:dtypes",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:array_ops_gen",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:math_ops_gen",
        "//machina/python/platform:client_testlib",
    ],
)

py_strict_library(
    name = "op_reg_gen",
    srcs = ["python/op_reg_gen.py"],
    deps = [
        "//machina:machina_py",
        "//machina/python/autograph/pyct:transformer",
        "//machina/python/autograph/pyct:transpiler",
        "//machina/python/framework:op_def_registry",
        "//machina/python/util:tf_inspect",
        "@pypi_gast//:pkg",
    ],
)

tf_py_strict_test(
    name = "op_reg_gen_test",
    size = "small",
    srcs = ["python/op_reg_gen_test.py"],
    deps = [
        ":composite",
        ":op_reg_gen",
        "//machina/compiler/mlir/python/mlir_wrapper:filecheck_wrapper",
        "//machina/python/platform:client_testlib",
    ],
)

py_strict_library(
    name = "test_utils",
    srcs = ["python/test_utils.py"],
    deps = [
        "//machina:machina_py",
        "//machina/python/eager:backprop",
        "//machina/python/framework:test_lib",
        "//machina/python/platform:client_testlib",
    ],
)

gen_op_libraries(
    name = "one_op",
    src = "define_op_template.py",
    deps = [
        "//machina/python/platform:flags",
        "@absl_py//absl:app",
    ],
)

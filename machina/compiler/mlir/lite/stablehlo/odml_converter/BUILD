load("@toolchain-project//mlir:tblgen.bzl", "gentbl_cc_library")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//machina:machina.bzl", "tf_cc_binary")
load("//machina:machina.default.bzl", "get_compatible_with_portable")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = [
        ":friends",
    ],
)

package_group(
    name = "friends",
    packages = [
        "//machina/compiler/mlir/lite/...",
    ],
)

tf_cc_binary(
    name = "odml-converter",
    srcs = ["odml_converter_main.cc"],
    compatible_with = get_compatible_with_portable(),
    visibility = [
        "//machina/compiler/mlir/lite/stablehlo/odml_converter:__subpackages__",
        "//third_party/odml/infra:__subpackages__",
    ],  # Prototype phase.
    deps = [
        ":shlo_simplify",
        "//machina/compiler/mlir:init_mlir",
        "//machina/compiler/mlir/lite:machina_lite",
        "//machina/compiler/mlir/lite/stablehlo:legalize_stablehlo_to_vhlo_pass",
        "//machina/compiler/mlir/machina:machina_types",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:ArithDialect",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:MlirOptLib",
        "@toolchain-project//mlir:Support",
        "@stablehlo//:chlo_ops",
        "@stablehlo//:stablehlo_ops",
    ],
)

cc_library(
    name = "shlo_simplify",
    srcs = [
        "transforms/shlo_simplify.cc",
    ],
    hdrs = ["passes.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":folders",
        ":passes_inc_gen",
        ":shlo_simplify_inc_gen",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:Support",
        "@toolchain-project//mlir:TransformUtils",
        "@stablehlo//:stablehlo_ops",
    ],
    alwayslink = 1,
)

gentbl_cc_library(
    name = "shlo_simplify_inc_gen",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = {"transforms/generated_shlo_simplify.inc": ["-gen-rewriters"]},
    tblgen = "@toolchain-project//mlir:mlir-tblgen",
    td_file = "transforms/shlo_simplify.td",
    deps = ["@stablehlo//:stablehlo_ops_td_files"],
)

cc_library(
    name = "folders",
    srcs = ["folders.cc"],
    hdrs = ["folders.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Support",
        "@stablehlo//:stablehlo_ops",
    ],
)

gentbl_cc_library(
    name = "passes_inc_gen",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = {"passes.h.inc": [
        "-gen-pass-decls",
        "-name=ODMLConverter",
    ]},
    tblgen = "@toolchain-project//mlir:mlir-tblgen",
    td_file = "passes.td",
    deps = ["@toolchain-project//mlir:PassBaseTdFiles"],
)

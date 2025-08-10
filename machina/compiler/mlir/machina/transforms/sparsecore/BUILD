load("@toolchain-project//mlir:tblgen.bzl", "gentbl_cc_library")
load("//machina:machina.default.bzl", "get_compatible_with_portable")
load("//machina/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = [
        "//machina/compiler/mlir:__subpackages__",
        "//machina/compiler/mlir/machina/transforms:__pkg__",
        "//machina/compiler/mlir/machina/transforms/host_runtime:__pkg__",
        "//machina/compiler/mlir/tf2xla/internal:__pkg__",
    ],
    licenses = ["notice"],
)

gentbl_cc_library(
    name = "sparsecore_passes_inc_gen",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = {"sparsecore_passes.h.inc": [
        "-gen-pass-decls",
        "-name=SparseCore",
    ]},
    tblgen = "@toolchain-project//mlir:mlir-tblgen",
    td_file = "sparsecore_passes.td",
    deps = [
        "@toolchain-project//mlir:PassBaseTdFiles",
    ],
)

cc_library(
    name = "sparsecore_passes",
    hdrs = [
        "sparsecore_passes.h",
    ],
    textual_hdrs = [
        "sparsecore_passes.h.inc",
    ],
    deps = [
        ":embedding_pipelining",
        ":embedding_program_key",
        ":embedding_sequencing",
        ":sparsecore_passes_inc_gen",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Pass",
    ],
)

cc_library(
    name = "embedding_pipelining",
    srcs = ["embedding_pipelining.cc"],
    hdrs = [
        "sparsecore_passes.h",
    ],
    deps = [
        ":sparsecore_passes_inc_gen",
        "//machina/compiler/jit:flags_headers",
        "//machina/compiler/mlir/machina",
        "//machina/compiler/mlir/machina:attribute_utils",
        "//machina/compiler/mlir/machina:machina_ops",
        "//machina/compiler/mlir/machina:machina_types",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:InliningUtils",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:Support",
        "@toolchain-project//mlir:TransformUtils",
    ],
)

cc_library(
    name = "embedding_sequencing",
    srcs = ["embedding_sequencing.cc"],
    hdrs = [
        "sparsecore_passes.h",
    ],
    deps = [
        ":sparsecore_passes_inc_gen",
        "//machina/compiler/jit:flags_headers",
        "//machina/compiler/mlir/machina",
        "//machina/compiler/mlir/machina:attribute_utils",
        "//machina/compiler/mlir/machina:machina_ops",
        "//machina/compiler/mlir/machina:machina_types",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:InliningUtils",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:Support",
        "@toolchain-project//mlir:TransformUtils",
    ],
)

cc_library(
    name = "embedding_program_key",
    srcs = ["embedding_program_key.cc"],
    hdrs = [
        "sparsecore_passes.h",
    ],
    deps = [
        ":sparsecore_passes_inc_gen",
        "//machina/compiler/mlir/machina",
        "//machina/compiler/mlir/machina:machina_ops",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:Support",
        "@local_xla//xla/mlir_hlo",
    ],
)

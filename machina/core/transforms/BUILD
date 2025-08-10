load("@toolchain-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//machina:machina.bzl", "tf_cc_binary", "tf_cc_test")
load("//machina:machina.default.bzl", "get_compatible_with_portable")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_compatible_with = get_compatible_with_portable(),
    default_visibility = [
        "//learning/brain/contrib/tpu_modeling/inference_converter_v2:__subpackages__",
        "//machina/core:__subpackages__",
        "//machina/tools/tfg_graph_transforms:__subpackages__",
    ],
    licenses = ["notice"],  # Apache 2.0
)

gentbl_cc_library(
    name = "PassIncGen",
    tbl_outs = {"passes.h.inc": [
        "-gen-pass-decls",
        "--name",
        "TFGraph",
    ]},
    tblgen = "@toolchain-project//mlir:mlir-tblgen",
    td_file = "passes.td",
    deps = [
        "@toolchain-project//mlir:PassBaseTdFiles",
    ],
)

cc_library(
    name = "utils",
    srcs = ["utils/utils.cc"],
    hdrs = ["utils/utils.h"],
    deps = [
        "//machina/core:framework",
        "//machina/core/ir:Dialect",
        "@com_google_absl//absl/strings",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:IR",
    ],
)

cc_library(
    name = "eval_utils",
    srcs = ["utils/eval_utils.cc"],
    hdrs = ["utils/eval_utils.h"],
    deps = [
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core/framework:node_def_proto_cc",
        "//machina/core/framework:tensor_proto_cc",
        "//machina/core/ir:Dialect",
        "//machina/core/ir/importexport:convert_tensor",
        "//machina/core/ir/importexport:graphdef_export",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Support",
    ],
)

gentbl_cc_library(
    name = "PDLLUtilsIncGen",
    tbl_outs = {"utils/pdll/PDLLUtils.h.inc": ["-x=cpp"]},
    tblgen = "@toolchain-project//mlir:mlir-pdll",
    td_file = "utils/pdll/utils.pdll",
    deps = [
        "//machina/core/ir:Dialect",
    ],
)

td_library(
    name = "PDLLUtilsFiles",
    srcs = ["utils/pdll/utils.pdll"],
    includes = ["include"],
)

cc_library(
    name = "pdll_utils",
    srcs = ["utils/pdll/utils.cc"],
    hdrs = ["utils/pdll/utils.h"],
    deps = [
        ":PDLLUtilsIncGen",
        ":utils",
        "//machina/core:framework_types_hdr",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Support",
    ],
)

cc_library(
    name = "op_cat_helper",
    srcs = ["utils/op_cat_helper.cc"],
    hdrs = ["utils/op_cat_helper.h"],
    deps = [
        "//machina/core:framework",
        "//machina/core/framework:op_def_proto_cc",
        "//machina/core/framework:types_proto_cc",
        "//machina/core/ir:Dialect",
        "@com_google_absl//absl/status",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Support",
    ],
)

cc_library(
    name = "PassRegistration",
    hdrs = ["pass_registration.h"],
    deps = [
        ":PassIncGen",
        "//machina/core/transforms/cf_sink:Pass",
        "//machina/core/transforms/consolidate_attrs:Pass",
        "//machina/core/transforms/const_dedupe_hoist:Pass",
        "//machina/core/transforms/constant_folding:Pass",
        "//machina/core/transforms/cse:Pass",
        "//machina/core/transforms/drop_unregistered_attribute:Pass",
        "//machina/core/transforms/eliminate_passthrough_iter_args:Pass",
        "//machina/core/transforms/func_to_graph:Pass",
        "//machina/core/transforms/functional_to_region:Pass",
        "//machina/core/transforms/graph_compactor:Pass",
        "//machina/core/transforms/graph_to_func:Pass",
        "//machina/core/transforms/legacy_call:Pass",
        "//machina/core/transforms/region_to_functional:Pass",
        "//machina/core/transforms/remapper:Pass",
        "//machina/core/transforms/shape_inference:Pass",
        "//machina/core/transforms/toposort:Pass",
    ],
)

cc_library(
    name = "graph_transform_wrapper",
    srcs = ["graph_transform_wrapper.cc"],
    hdrs = ["graph_transform_wrapper.h"],
    deps = [
        "//machina/compiler/mlir/machina:error_util",
        "//machina/core:core_cpu_base",
        "//machina/core/framework:graph_debug_info_proto_cc",
        "//machina/core/ir/importexport:graphdef_export",
        "//machina/core/ir/importexport:graphdef_import",
        "//machina/core/platform:status",
        "//machina/core/platform:statusor",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Pass",
    ],
)

tf_cc_test(
    name = "graph_transform_wrapper_test",
    srcs = ["graph_transform_wrapper_test.cc"],
    deps = [
        ":graph_transform_wrapper",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core/common_runtime:graph_constructor",
        "//machina/core/ir:Dialect",
        "//machina/core/platform:status",
        "@com_google_absl//absl/strings",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:IR",
    ],
)

tf_cc_test(
    name = "eval_utils_test",
    size = "small",
    srcs = ["utils/eval_utils_test.cc"],
    deps = [
        ":eval_utils",
        "//machina/core:all_kernels",
        "//machina/core:framework",
        "//machina/core:ops",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core/ir:Dialect",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Parser",
        "@toolchain-project//mlir:Support",
    ],
)

# Custom `mlir-opt` replacement that links our dialect and passes
tf_cc_binary(
    name = "tfg-transforms-opt",
    srcs = ["tfg-transforms-opt.cc"],
    deps = [
        ":PassRegistration",
        "//machina/core:all_kernels",
        "//machina/core:ops",
        "//machina/core/ir:Dialect",
        "//machina/core/ir:tf_op_registry",
        "//machina/core/ir/types:Dialect",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:MlirOptLib",
        "@toolchain-project//mlir:Transforms",
    ],
)

filegroup(
    name = "test_utilities",
    testonly = True,
    data = [
        ":tfg-transforms-opt",
        "@toolchain-project//toolchain:FileCheck",
        "@toolchain-project//toolchain:not",
    ],
)

# Description:
#   Utilities to perform MLIR TFG graph transformations.

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load(
    "//machina:machina.bzl",
    "tf_cc_binary",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    licenses = ["notice"],
)

cc_library(
    name = "utils",
    srcs = [
        "utils.cc",
    ],
    hdrs = [
        "utils.h",
    ],
    deps = [
        "//machina/cc/saved_model/image_format:internal_api",
        "//machina/core:lib",
        "//machina/core/platform:errors",
        "//machina/core/platform:path",
        "//machina/core/platform:status",
        "//machina/core/protobuf:for_core_protos_cc",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
    ],
)

TFG_GRAPH_TRANSFORM_DEPS = [
    ":utils",
    "@toolchain-project//toolchain:Support",
    "@toolchain-project//mlir:IR",
    "@toolchain-project//mlir:Pass",
    "@toolchain-project//mlir:Transforms",
    "//machina/compiler/mlir:init_mlir",
    "//machina/compiler/mlir/machina",
    "//machina/compiler/mlir/machina:error_util",
    "//machina/core:lib",
    "//machina/core/ir:Dialect",
    "//machina/core/protobuf:for_core_protos_cc",
    "//machina/core/transforms:PassRegistration",
    "//machina/compiler/tf2xla/ops:xla_ops",
    "//machina/core:ops",
    "//machina/core:protos_all_cc",
    "//machina/core/ir/importexport:graphdef_export",
    "//machina/core/ir/importexport:graphdef_import",
    "//machina/core/ir/importexport:savedmodel_export",
    "//machina/core/ir/importexport:savedmodel_import",
    "//machina/core/ir:tf_op_registry",
]

# Description:
#   A tool that provides a mechanism to run TFG graph optimizations operating on the
#   GraphDef from the supplied SavedModel as an input.
tf_cc_binary(
    name = "tfg_graph_transforms",
    srcs = [
        "tfg_graph_transforms_main.cc",
    ],
    deps = TFG_GRAPH_TRANSFORM_DEPS + [
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status:statusor",
        "@toolchain-project//mlir:Support",
    ],
)

# Description:
#    The tool wrapped into a library, so that it could be used
#    when custom defined ops and passes need to be added.
cc_library(
    name = "tfg_graph_transforms_main",
    srcs = [
        "tfg_graph_transforms_main.cc",
    ],
    visibility = ["//visibility:public"],
    deps = TFG_GRAPH_TRANSFORM_DEPS + [
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status:statusor",
        "@toolchain-project//mlir:Support",
    ],
)

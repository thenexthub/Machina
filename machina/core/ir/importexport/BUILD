load("//machina:machina.bzl", "tf_cc_binary")
load("//machina/compiler/mlir:glob_lit_test.bzl", "glob_lit_tests")
load("//machina/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = [
        ":__subpackages__",
        ":friends",
        "//machina/compiler/mlir:__subpackages__",
        "//machina/core:__subpackages__",
        "//machina/core:dependency_allowlist",
        "//machina/tools/tfg_graph_transforms:__subpackages__",
    ],
    licenses = ["notice"],  # Apache 2.0
)

package_group(
    name = "friends",
    packages = [
        "//learning/brain/contrib/tpu_modeling/inference_converter_v2/mlir/...",
        "//waymo/ml/cn/...",
        "//waymo/ml/compiler/mlir/...",
    ],
)

cc_library(
    name = "parse_text_proto",
    srcs = [
        "parse_text_proto.cc",
    ],
    hdrs = ["parse_text_proto.h"],
    deps = [
        "//machina/core:framework_lite",
        "//machina/core:lib_proto_parsing",
        "//machina/core/platform:casts",
        "@com_google_absl//absl/strings",
        "@com_google_protobuf//:protobuf",
    ],
)

cc_library(
    name = "mangling",
    srcs = [
        "mangling.cc",
    ],
    hdrs = ["mangling.h"],
    deps = [
        ":parse_text_proto",
        "//machina/core:framework",
        "//machina/core:framework_lite",
        "//machina/core:lib_proto_parsing",
        "//machina/core:protos_all_cc",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "load_proto",
    srcs = [
        "load_proto.cc",
    ],
    hdrs = ["load_proto.h"],
    deps = [
        ":parse_text_proto",
        "//machina/core:framework_lite",
        "//machina/core:lib_proto_parsing",
        "@com_google_absl//absl/strings",
        "@toolchain-project//toolchain:Support",
    ],
)

cc_library(
    name = "convert_types",
    srcs = [
        "convert_types.cc",
    ],
    hdrs = ["convert_types.h"],
    deps = [
        "//machina/core:framework",
        "//machina/core:lib_proto_parsing",
        "//machina/core:protos_all_cc",
        "//machina/core/ir:Dialect",
        "//machina/core/ir/types:Dialect",
        "//machina/core/platform:statusor",
        "@com_google_absl//absl/strings",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Support",
    ],
)

cc_library(
    name = "convert_tensor",
    srcs = [
        "convert_tensor.cc",
    ],
    hdrs = ["convert_tensor.h"],
    deps = [
        ":convert_types",
        ":mangling",
        "//machina/core:framework",
        "//machina/core:framework_lite",
        "//machina/core:lib_proto_parsing",
        "//machina/core:protos_all_cc",
        "//machina/core:tflite_portable_logging",
        "//machina/core/ir:Dialect",
        "//machina/core/ir/types:Dialect",
        "//machina/core/platform:errors",
        "//machina/core/platform:statusor",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Support",
        "@local_tsl//tsl/platform:ml_dtypes",
    ],
)

cc_library(
    name = "savedmodel_import",
    srcs = ["savedmodel_import.cc"],
    hdrs = ["savedmodel_import.h"],
    deps = [
        ":graphdef_import",
        "//machina/core:core_cpu_base",
        "//machina/core:protos_all_cc",
        "//machina/core/platform:errors",
        "//machina/core/platform:statusor",
        "@toolchain-project//mlir:IR",
    ],
)

cc_library(
    name = "savedmodel_export",
    srcs = ["savedmodel_export.cc"],
    hdrs = ["savedmodel_export.h"],
    deps = [
        ":graphdef_export",
        "//machina/core:core_cpu_base",
        "//machina/core:protos_all_cc",
        "//machina/core/platform:errors",
        "//machina/core/platform:status",
        "@toolchain-project//mlir:IR",
    ],
)

cc_library(
    name = "graphdef_import",
    srcs = [
        "functiondef_import.cc",
        "graphdef_import.cc",
    ],
    hdrs = [
        "functiondef_import.h",
        "graphdef_import.h",
    ],
    deps = [
        ":convert_attributes",
        ":convert_types",
        "//machina/core:core_cpu_base",
        "//machina/core:framework",
        "//machina/core:protos_all_cc",
        "//machina/core/ir:Dialect",
        "//machina/core/ir/types:Dialect",
        "//machina/core/platform:errors",
        "//machina/core/platform:status",
        "//machina/core/platform:statusor",
        "//machina/core/platform:stringpiece",
        "@com_google_absl//absl/container:flat_hash_map",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Support",
    ],
)

cc_library(
    name = "graphdef_export",
    srcs = [
        "functiondef_export.cc",
        "graphdef_export.cc",
    ],
    hdrs = [
        "functiondef_export.h",
        "graphdef_export.h",
    ],
    deps = [
        ":convert_attributes",
        ":convert_types",
        "//machina/core:core_cpu_base",
        "//machina/core:framework",
        "//machina/core:protos_all_cc",
        "//machina/core/ir:Dialect",
        "//machina/core/ir/types:Dialect",
        "//machina/core/platform:errors",
        "//machina/core/platform:status",
        "//machina/core/platform:statusor",
        "@com_google_absl//absl/strings",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Support",
    ],
)

cc_library(
    name = "convert_attributes",
    srcs = [
        "convert_attributes.cc",
    ],
    hdrs = ["convert_attributes.h"],
    deps = [
        ":convert_tensor",
        ":convert_types",
        ":mangling",
        "//machina/core:framework",
        "//machina/core:protos_all_cc",
        "//machina/core/ir:Dialect",
        "//machina/core/ir/types:Dialect",
        "//machina/core/platform:errors",
        "//machina/core/platform:statusor",
        "@com_google_absl//absl/strings",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Support",
        "@local_xla//xla:status_macros",
    ],
)

# Command line tool to convert to/from GraphDef to MLIR Graph.
tf_cc_binary(
    name = "tfg-translate",
    srcs = ["tfg-translate.cc"],
    deps = [
        ":graphdef_export",
        ":graphdef_import",
        ":load_proto",
        "//machina/compiler/mlir:init_mlir",
        "//machina/core:ops",  # Ops need to be registered for import.
        "//machina/core/framework:graph_debug_info_proto_cc",
        "//machina/core/framework:graph_proto_cc",
        "//machina/core/ir:Dialect",
        "//machina/core/platform:status",
        "@com_google_absl//absl/log",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Support",
        "@toolchain-project//mlir:TranslateLib",
    ],
)

glob_lit_tests(
    name = "all_tests",
    data = ["//machina/core/ir:test_utilities"],
    driver = "//machina/core/ir:run_lit.sh",
    exclude = [],
    test_file_exts = ["mlir"],
)

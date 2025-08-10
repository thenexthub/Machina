load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "parsers",
    srcs = ["parsers.cc"],
    hdrs = ["parsers.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//machina/compiler/mlir/machina:mlir_roundtrip_flags",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core/platform:types",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@toolchain-project//toolchain:Support",
        "@local_xla//xla:status_macros",
    ],
)

cc_library(
    name = "file_tf_mlir_translate",
    srcs = ["file_tf_mlir_translate.cc"],
    hdrs = ["file_tf_mlir_translate.h"],
    deps = [
        ":parsers",
        "//machina/compiler/mlir/machina:import_utils",
        "//machina/compiler/mlir/machina:mlir_roundtrip_flags",
        "//machina/compiler/mlir/tf2xla/api/v2:graph_to_tf_executor",
        "//machina/core:core_cpu_base",
        "//machina/core:framework",
        "//machina/core:protos_all_cc",
        "//machina/core/framework:graph_proto_cc",
        "//machina/core/grappler/utils:transitive_fanin",
        "//machina/core/platform:byte_order",
        "//machina/core/platform:errors",
        "//machina/core/protobuf:for_core_protos_cc",
        "//machina/core/util/tensor_bundle:byteswaptensor",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Parser",
        "@toolchain-project//mlir:Support",
        "@local_xla//xla/tsl/platform:errors",
        "@local_xla//xla/tsl/platform:statusor",
    ],
)

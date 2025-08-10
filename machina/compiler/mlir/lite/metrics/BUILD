# Placeholder: load py_proto_library
load("//machina:machina.bzl", "tf_cc_test")
load(
    "//machina/core/platform:build_config.bzl",
    "tf_proto_library",
)
load("//machina/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = [
        "//machina:__pkg__",
        "//machina/compiler/mlir/lite:__subpackages__",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "error_collector",
    srcs = ["error_collector.cc"],
    hdrs = ["error_collector.h"],
    deps = [
        ":converter_error_data_proto_cc",
        ":types_util",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "error_collector_inst",
    srcs = ["error_collector_inst.cc"],
    hdrs = ["error_collector_inst.h"],
    deps = [
        ":converter_error_data_proto_cc",
        ":error_collector",
        ":types_util",
        "@com_google_absl//absl/strings",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:Support",
    ],
)

tf_cc_test(
    name = "error_collector_inst_test",
    srcs = ["error_collector_inst_test.cc"],
    data = [
        "testdata/strided_slice.mlir",
    ],
    deps = [
        ":converter_error_data_proto_cc",
        ":error_collector",
        ":error_collector_inst",
        ":types_util",
        "//machina/compiler/mlir/machina",
        "//machina/core:test",
        "//machina/core/platform:errors",
        "//machina/core/platform:resource_loader",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest_main",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Parser",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:Support",
        "@local_xla//xla/tsl/platform:statusor",
    ],
)

cc_library(
    name = "types_util",
    srcs = ["types_util.cc"],
    hdrs = ["types_util.h"],
    deps = [
        ":converter_error_data_proto_cc",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Support",
    ],
)

tf_proto_library(
    name = "converter_error_data_proto",
    srcs = ["converter_error_data.proto"],
    visibility = [
        "//visibility:public",
    ],
)

# copybara:uncomment_begin(google-only)
# py_proto_library(
#     name = "converter_error_data_proto_py",
#     visibility = [
#         "//visibility:public",
#     ],
#     deps = [":converter_error_data_proto"],
# )
# copybara:uncomment_end

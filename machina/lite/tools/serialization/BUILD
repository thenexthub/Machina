load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = [
        "//visibility:public",
    ],
    features = ["-parse_headers"],
    licenses = ["notice"],
)

cc_binary(
    name = "option_writer_generator",
    srcs = ["option_writer_generator.cc"],
    deps = [
        "//machina/lite/schema:schema_fbs_with_reflection",
        "@flatbuffers",
    ],
)

cc_library(
    name = "writer_lib_enum",
    hdrs = ["enum_mapping.h"],
    deps = [
        "//machina/compiler/mlir/lite/schema:schema_fbs_with_mutable",
        "//machina/lite:builtin_op_data",
    ],
)

cc_library(
    name = "writer_lib",
    srcs = [
        "writer_lib.cc",
    ],
    hdrs = [
        "writer_lib.h",
    ],
    data = [
        ":option_writer_gen",
    ],
    textual_hdrs = ["option_writer_generated.h"],
    deps = [
        "//machina/compiler/mlir/lite/schema:schema_conversion_utils",
        "//machina/compiler/mlir/lite/schema:schema_fbs_with_mutable",
        "//machina/compiler/mlir/lite/tools/versioning",
        "//machina/lite:builtin_op_data",
        "//machina/lite:framework",
        "//machina/lite:schema_fbs_version",
        "//machina/lite/core:framework_stable",
        "//machina/lite/core:model_builder",
        "//machina/lite/core/c:common",
        "//machina/lite/schema:schema_conversion_utils",
        "//machina/lite/tools/serialization:writer_lib_enum",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@flatbuffers//:runtime_cc",
    ],
)

cc_binary(
    name = "writer",
    srcs = ["writer.cc"],
    deps = [
        ":writer_lib",
        "//machina/lite:framework",
        "//machina/lite/core:framework",
        "//machina/lite/core/kernels:builtin_ops",
    ],
)

cc_binary(
    name = "writer_test",
    srcs = ["writer_test.cc"],
    deps = [
        ":writer_lib",
        "//machina/lite:framework",
        "//machina/lite/c:c_api_types",
        "//machina/lite/core:framework",
        "//machina/lite/core/kernels:builtin_ops",
    ],
)

cc_test(
    name = "writer_lib_test",
    size = "small",
    srcs = ["writer_lib_test.cc"],
    deps = [
        ":writer_lib",
        "//machina/compiler/mlir/lite/schema:schema_fbs_with_mutable",
        "//machina/lite:builtin_ops",
        "//machina/lite:framework",
        "//machina/lite:util",
        "//machina/lite/core:framework",
        "//machina/lite/core/c:c_api_types",
        "//machina/lite/core/c:common",
        "//machina/lite/core/kernels:builtin_ops",
        "//machina/lite/kernels:subgraph_test_util",
        "//machina/lite/schema:schema_fbs",
        "@com_google_absl//absl/log:check",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:logging",
    ],
)

genrule(
    name = "option_writer_gen",
    outs = ["option_writer_generated.h"],
    cmd = "$(location :option_writer_generator) $(@)",
    tools = [":option_writer_generator"],
)

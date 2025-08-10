load("@rules_cc//cc:cc_library.bzl", "cc_library")
load(
    "//machina:machina.bzl",
    "tf_cc_test",
)
load("//machina:machina.default.bzl", "get_compatible_with_portable")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    licenses = ["notice"],
)

cc_library(
    name = "operator",
    srcs = [
        "operator.cc",
    ],
    hdrs = [
        "builtin_operator.h",
        "custom_operator.h",
        "operator.h",
        "simple_operator.h",
    ],
    compatible_with = get_compatible_with_portable(),
    visibility = [
        "//machina/lite/toco:__subpackages__",
    ],
    deps = [
        ":types",
        "//machina/compiler/mlir/lite/delegates/flex:allowlisted_flex_ops_lib",
        "//machina/compiler/mlir/lite/tools/versioning",
        "//machina/compiler/mlir/lite/tools/versioning:op_signature",
        "//machina/core:framework",
        "//machina/core:protos_all_cc",
        "//machina/lite/c:c_api_types",
        "//machina/lite/core/c:common",
        "//machina/lite/schema:schema_fbs",
        "//machina/lite/toco:graph_transformations",
        "//machina/lite/toco:model",
        "//machina/lite/toco:runtime",
        "//machina/lite/toco:toco_port",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/memory",
        "@flatbuffers",
    ],
)

tf_cc_test(
    name = "operator_test",
    srcs = [
        "operator_test.cc",
    ],
    deps = [
        ":operator",
        "//machina/core:ops",
        "//machina/core:protos_all_cc",
        "//machina/lite/schema:schema_fbs",
        "//machina/lite/toco:model",
        "//machina/lite/toco:runtime",
        "//machina/lite/toco:tooling_util",
        "@com_google_absl//absl/log:check",
        "@com_google_googletest//:gtest_main",
        "@flatbuffers",
    ],
)

cc_library(
    name = "types",
    srcs = [
        "types.cc",
    ],
    hdrs = [
        "types.h",
    ],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//machina/lite:string_util",
        "//machina/lite/schema:schema_fbs",
        "//machina/lite/toco:model",
        "//machina/lite/toco:runtime",
        "@com_google_absl//absl/log",
        "@flatbuffers//:runtime_cc",
    ],
)

tf_cc_test(
    name = "types_test",
    srcs = [
        "types_test.cc",
    ],
    deps = [
        ":types",
        "//machina/core:ops",
        "//machina/lite/schema:schema_fbs",
        "//machina/lite/toco:model",
        "//machina/lite/toco:runtime",
        "@com_google_googletest//:gtest_main",
        "@flatbuffers//:runtime_cc",
    ],
)

cc_library(
    name = "export",
    srcs = [
        "export.cc",
    ],
    hdrs = [
        "export.h",
    ],
    compatible_with = get_compatible_with_portable(),
    visibility = ["//visibility:public"],
    deps = [
        ":operator",
        ":types",
        "//machina/compiler/mlir/lite/quantization/lite/toco_legacy:quantize_weights",
        "//machina/compiler/mlir/lite/schema:schema_conversion_utils",
        "//machina/compiler/mlir/lite/tools/versioning",
        "//machina/core:lib_proto_parsing",
        "//machina/core/platform:status",
        "//machina/lite:schema_fbs_version",
        "//machina/lite:util",
        "//machina/lite/c:c_api_types",
        "//machina/lite/schema:schema_fbs",
        "//machina/lite/toco:model",
        "//machina/lite/toco:toco_port",
        "//machina/lite/toco:tooling_util",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
        "@flatbuffers",
    ],
)

tf_cc_test(
    name = "export_test",
    srcs = [
        "export_test.cc",
    ],
    deps = [
        ":export",
        ":operator",
        ":types",
        "//machina/compiler/mlir/lite/schema:schema_utils",
        "//machina/core:lib_proto_parsing",
        "//machina/core:ops",
        "//machina/core:protos_all_cc",
        "//machina/lite/schema:schema_fbs",
        "//machina/lite/toco:model",
        "@com_google_absl//absl/log",
        "@com_google_googletest//:gtest_main",
        "@flatbuffers",
        "@local_xla//xla/tsl/protobuf:error_codes_proto_impl_cc",
    ],
)

cc_library(
    name = "import",
    srcs = [
        "import.cc",
    ],
    hdrs = [
        "import.h",
    ],
    compatible_with = get_compatible_with_portable(),
    visibility = ["//visibility:public"],
    deps = [
        ":operator",
        ":types",
        "//machina/compiler/mlir/lite/schema:schema_utils",
        "//machina/lite:framework",
        "//machina/lite/core:framework",
        "//machina/lite/core/tools:verifier",
        "//machina/lite/schema:schema_fbs",
        "//machina/lite/toco:model",
        "//machina/lite/toco:model_flags_proto_cc",
        "//machina/lite/toco:tooling_util",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@flatbuffers",
    ],
)

tf_cc_test(
    name = "import_test",
    srcs = [
        "import_test.cc",
    ],
    tags = [
        "no_oss",  # TODO(b/273558651): Enable after updating flatbuffer version.
    ],
    deps = [
        ":import",
        "//machina/compiler/mlir/lite/schema:schema_conversion_utils",
        "//machina/core:ops",
        "//machina/lite:schema_fbs_version",
        "//machina/lite/schema:schema_fbs",
        "//machina/lite/toco:model",
        "//machina/lite/toco:model_flags_proto_cc",
        "//machina/lite/toco:toco_port",
        "@com_google_googletest//:gtest_main",
        "@flatbuffers",
    ],
)

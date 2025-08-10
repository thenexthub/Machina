# Description:
#   C++ implementation code for the summary writing APIs.

load(
    "//machina:machina.bzl",
    "tf_cc_binary",
    "tf_cc_test",
    "tf_copts",
)
load("//machina/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//machina:internal"],
    licenses = ["notice"],
)

cc_library(
    name = "schema",
    srcs = ["schema.cc"],
    hdrs = ["schema.h"],
    copts = tf_copts(),
    deps = [
        "//machina/core:lib",
        "//machina/core/lib/db:sqlite",
        "@com_google_absl//absl/status",
    ],
)

tf_cc_test(
    name = "schema_test",
    size = "small",
    srcs = ["schema_test.cc"],
    deps = [
        ":schema",
        "//machina/core:test",
        "//machina/core:test_main",
    ],
)

cc_library(
    name = "summary_db_writer",
    srcs = ["summary_db_writer.cc"],
    hdrs = ["summary_db_writer.h"],
    copts = tf_copts(),
    deps = [
        ":summary_converter",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core:protos_all_cc",
        "//machina/core/kernels:summary_interface",
        "//machina/core/lib/db:sqlite",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@local_xla//xla/tsl/protobuf:error_codes_proto_impl_cc",
        "@local_xla//xla/tsl/protobuf:histogram_proto_cc",
    ],
)

tf_cc_test(
    name = "summary_db_writer_test",
    size = "small",
    srcs = ["summary_db_writer_test.cc"],
    deps = [
        ":schema",
        ":summary_db_writer",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core/lib/db:sqlite",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@local_xla//xla/tsl/protobuf:histogram_proto_cc",
    ],
)

cc_library(
    name = "summary_file_writer",
    srcs = ["summary_file_writer.cc"],
    hdrs = ["summary_file_writer.h"],
    copts = tf_copts(),
    deps = [
        ":summary_converter",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core:protos_all_cc",
        "//machina/core/kernels:summary_interface",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@local_xla//xla/tsl/protobuf:error_codes_proto_impl_cc",
    ],
)

tf_cc_test(
    name = "summary_file_writer_test",
    size = "medium",  # file i/o
    timeout = "short",
    srcs = ["summary_file_writer_test.cc"],
    deps = [
        ":summary_file_writer",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core:protos_all_cc",
        "//machina/core:test",
        "//machina/core:test_main",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "summary_converter",
    srcs = ["summary_converter.cc"],
    hdrs = ["summary_converter.h"],
    copts = tf_copts(),
    visibility = ["//visibility:private"],
    deps = [
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core:protos_all_cc",
        "//machina/core/lib/png:png_io",
        "@com_google_absl//absl/status",
    ],
)

tf_cc_binary(
    name = "loader",
    srcs = ["loader.cc"],
    linkstatic = 1,
    deps = [
        ":schema",
        ":summary_db_writer",
        "//machina/core:framework",
        "//machina/core:framework_internal",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core/lib/db:sqlite",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@local_xla//xla/tsl/protobuf:error_codes_proto_impl_cc",
    ],
)

tf_cc_binary(
    name = "vacuum",
    srcs = ["vacuum.cc"],
    linkstatic = 1,
    deps = [
        "//machina/core:framework_internal",
        "//machina/core:lib",
        "//machina/core/lib/db:sqlite",
        "@com_google_absl//absl/log",
    ],
)

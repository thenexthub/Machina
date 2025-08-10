# Utilities for building XLA computations.

load("//machina:machina.default.bzl", "filegroup")
load("//machina/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//machina/compiler/tf2xla:friends"],
    licenses = ["notice"],
)

# Filegroup used to collect source files for dependency checking.
filegroup(
    name = "c_srcs",
    data = glob([
        "**/*.cc",
        "**/*.h",
    ]),
)

cc_library(
    name = "broadcast",
    srcs = ["broadcast.cc"],
    hdrs = ["broadcast.h"],
    deps = [
        "//machina/compiler/tf2xla:common",
        "//machina/core:framework",
        "//machina/core/platform:errors",
        "//machina/core/platform:status",
        "//machina/core/platform:statusor",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@local_xla//xla/hlo/builder:xla_builder",
        "@local_xla//xla/hlo/builder/lib:broadcast",
        "@local_xla//xla/tsl/platform:errors",
        "@local_xla//xla/tsl/platform:statusor",
    ],
)

cc_library(
    name = "random",
    srcs = ["random.cc"],
    hdrs = ["random.h"],
    deps = [
        "//machina/compiler/tf2xla:xla_compiler",
        "//machina/compiler/tf2xla:xla_helpers",
        "//machina/core:protos_all_cc",
        "//machina/core/platform:statusor",
        "@local_xla//xla:xla_data_proto_cc",
        "@local_xla//xla/hlo/builder:xla_builder",
        "@local_xla//xla/hlo/builder/lib:constants",
        "@local_xla//xla/hlo/builder/lib:math",
    ],
)

cc_library(
    name = "scatter",
    srcs = ["scatter.cc"],
    hdrs = ["scatter.h"],
    deps = [
        "//machina/core:lib",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@local_xla//xla:shape_util",
        "@local_xla//xla:status_macros",
        "@local_xla//xla:xla_data_proto_cc",
        "@local_xla//xla/hlo/builder:xla_builder",
        "@local_xla//xla/hlo/builder:xla_computation",
    ],
)

cc_library(
    name = "util",
    srcs = ["util.cc"],
    hdrs = ["util.h"],
    deps = [
        "//machina/core:lib",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/types:span",
        "@local_xla//xla:literal",
        "@local_xla//xla:literal_util",
        "@local_xla//xla:shape_util",
        "@local_xla//xla:xla_data_proto_cc",
        "@local_xla//xla/hlo/builder:xla_builder",
        "@local_xla//xla/hlo/builder:xla_computation",
    ],
)

cc_library(
    name = "data_format",
    srcs = ["data_format.cc"],
    hdrs = ["data_format.h"],
    deps = [
        "//machina/core:framework",
        "//machina/core:lib",
        "@com_google_absl//absl/status:statusor",
        "@local_xla//xla:shape_util",
        "@local_xla//xla:util",
        "@local_xla//xla/hlo/builder:xla_builder",
    ],
)

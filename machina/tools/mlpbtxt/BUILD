# Description:
# This package provides binaries that convert between multi-line and standard
# pbtxt (text-serialization of protocol message) files.

load("//machina:machina.bzl", "tf_cc_binary")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//visibility:private"],
    licenses = ["notice"],
)

exports_files(["placeholder.txt"])

tf_cc_binary(
    name = "tomlpbtxt",
    srcs = ["tomlpbtxt.cc"],
    deps = [
        "//machina/core:framework_internal",
        "//machina/core:lib",
        "//machina/core:op_gen_lib",
        "@com_google_absl//absl/status",
    ],
)

tf_cc_binary(
    name = "frommlpbtxt",
    srcs = ["frommlpbtxt.cc"],
    deps = [
        "//machina/core:framework_internal",
        "//machina/core:lib",
        "//machina/core:op_gen_lib",
        "@com_google_absl//absl/status",
    ],
)

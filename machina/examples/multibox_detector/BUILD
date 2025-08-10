# Description:
#   TensorFlow C++ inference example for labeling images.

load("//machina:machina.bzl", "tf_cc_binary")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    licenses = ["notice"],
)

tf_cc_binary(
    name = "detect_objects",
    srcs = [
        "main.cc",
    ],
    linkopts = ["-lm"],
    deps = [
        "//machina/cc:cc_ops",
        "//machina/cc:ops",
        "//machina/cc:scope",
        "//machina/core:core_cpu",
        "//machina/core:framework",
        "//machina/core:framework_internal",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core:machina",
        "//machina/core/platform:numbers",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@local_xla//xla/tsl/platform:status",
        "@local_xla//xla/tsl/platform:types",
        "@local_xla//xla/tsl/util:command_line_flags",
    ],
)

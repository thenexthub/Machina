load("//machina:machina.default.bzl", "filegroup")
load("//machina/core/platform:rules_cc.bzl", "cc_library")

# Experimental ops. These will eventually be replaced by machine-generated versions.
package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    licenses = ["notice"],
)

cc_library(
    name = "array_ops",
    srcs = [
        "array_ops.cc",
    ],
    hdrs = [
        "array_ops.h",
    ],
    visibility = [
        "//machina:internal",
    ],
    deps = [
        "//machina/c/eager:abstract_context",
        "//machina/c/eager:abstract_operation",
        "//machina/c/eager:abstract_tensor_handle",
        "//machina/c/eager:tracing_utils",
        "//machina/core:protos_all_cc",
        "//machina/core/platform:status",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:span",
        "@local_xla//xla/tsl/platform:errors",
    ],
)

cc_library(
    name = "io_ops",
    srcs = [
        "io_ops.cc",
    ],
    hdrs = [
        "io_ops.h",
    ],
    visibility = [
        "//machina:internal",
    ],
    deps = [
        "//machina/c/eager:abstract_context",
        "//machina/c/eager:abstract_operation",
        "//machina/c/eager:abstract_tensor_handle",
        "//machina/c/eager:tracing_utils",
        "//machina/core:protos_all_cc",
        "//machina/core/platform:status",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:span",
        "@local_xla//xla/tsl/platform:errors",
    ],
)

cc_library(
    name = "math_ops",
    srcs = [
        "math_ops.cc",
    ],
    hdrs = [
        "math_ops.h",
    ],
    visibility = [
        "//machina:internal",
    ],
    deps = [
        "//machina/c/eager:abstract_context",
        "//machina/c/eager:abstract_operation",
        "//machina/c/eager:abstract_tensor_handle",
        "//machina/c/eager:tracing_utils",
        "//machina/core/platform:status",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:span",
        "@local_xla//xla/tsl/platform:errors",
    ],
)

cc_library(
    name = "nn_ops",
    srcs = [
        "nn_ops.cc",
    ],
    hdrs = [
        "nn_ops.h",
    ],
    visibility = [
        "//machina:internal",
    ],
    deps = [
        "//machina/c/eager:abstract_context",
        "//machina/c/eager:abstract_operation",
        "//machina/c/eager:abstract_tensor_handle",
        "//machina/c/eager:tracing_utils",
        "//machina/core/platform:status",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:span",
        "@local_xla//xla/tsl/platform:errors",
    ],
)

cc_library(
    name = "resource_variable_ops",
    srcs = [
        "resource_variable_ops.cc",
    ],
    hdrs = [
        "resource_variable_ops.h",
    ],
    visibility = [
        "//machina:internal",
    ],
    deps = [
        "//machina/c/eager:abstract_context",
        "//machina/c/eager:abstract_operation",
        "//machina/c/eager:abstract_tensor_handle",
        "//machina/c/eager:tracing_utils",
        "//machina/core:framework",
        "//machina/core:portable_gif_internal",
        "//machina/core:protos_all_cc",
        "//machina/core/platform:status",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:span",
        "@local_xla//xla/tsl/platform:errors",
    ],
)

cc_library(
    name = "ops",
    hdrs = [
        "array_ops.h",
        "io_ops.h",
        "math_ops.h",
        "nn_ops.h",
        "resource_variable_ops.h",
    ],
    visibility = [
        "//machina:internal",
    ],
    deps = [
        ":array_ops",
        ":io_ops",
        ":math_ops",
        ":nn_ops",
        ":resource_variable_ops",
        "//machina/c/eager:abstract_context",
        "//machina/c/eager:abstract_tensor_handle",
        "//machina/core:framework",
        "//machina/core:portable_gif_internal",
        "//machina/core:protos_all_cc",
        "//machina/core/platform:status",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:span",
    ],
)

filegroup(
    name = "pywrap_required_hdrs",
    srcs = [
        "array_ops.h",
        "io_ops.h",
        "math_ops.h",
        "nn_ops.h",
        "resource_variable_ops.h",
    ],
    visibility = ["//machina/python:__subpackages__"],
)

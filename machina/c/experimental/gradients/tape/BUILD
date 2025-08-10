# A tape built on top of unified execution APIs.
load("//machina/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    licenses = ["notice"],
)

cc_library(
    name = "tape_context",
    srcs = ["tape_context.cc"],
    hdrs = [
        "tape_context.h",
    ],
    visibility = [
        "//machina:internal",
    ],
    deps = [
        ":tape_operation",
        "//machina/c/eager:abstract_context",
        "//machina/c/eager:abstract_function",
        "//machina/c/eager:gradients_internal",
        "//machina/core:portable_gif_internal",
        "//machina/core/platform:status",
        "@com_google_absl//absl/status",
    ],
)

cc_library(
    name = "tape_operation",
    srcs = ["tape_operation.cc"],
    hdrs = [
        "tape_operation.h",
    ],
    visibility = [
        "//machina:internal",
    ],
    deps = [
        "//machina/c:tensor_interface",
        "//machina/c/eager:abstract_context",
        "//machina/c/eager:abstract_operation",
        "//machina/c/eager:abstract_tensor_handle",
        "//machina/c/eager:gradients_internal",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:portable_gif_internal",
        "//machina/core:protos_all_cc",
        "//machina/core/platform:errors",
        "//machina/core/platform:status",
        "//machina/core/platform:strcat",
        "//machina/core/platform:stringpiece",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:span",
        "@local_xla//xla/tsl/platform:errors",
    ],
)

cc_library(
    name = "tape",
    hdrs = [
        "tape_context.h",
        "tape_operation.h",
    ],
    visibility = [
        "//machina:internal",
    ],
    deps = [
        ":tape_context",
        ":tape_operation",
        "//machina/c:tensor_interface",
        "//machina/c/eager:abstract_context",
        "//machina/c/eager:abstract_function",
        "//machina/c/eager:abstract_operation",
        "//machina/c/eager:abstract_tensor_handle",
        "//machina/c/eager:gradients_internal",
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
        "tape_context.h",
        "tape_operation.h",
    ],
    visibility = [
        "//machina:internal",
    ],
)

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load(
    "//machina:machina.bzl",
    "tf_cc_binary",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "generate_cpp_main",
    srcs = ["generate_cpp_main.cc"],
    deps = [
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "//machina/c/experimental/ops/gen/common",
        "//machina/c/experimental/ops/gen/cpp/renderers",
        "//machina/core:framework",
        "//machina/core:framework_internal",
        "//machina/c/experimental/ops/gen/cpp",
        "//machina/c/experimental/ops/gen/model",
        "//machina/core:lib",
        # Without this line to link in ops, the global registry will be empty:
        "//machina/core:ops",
    ],
)

tf_cc_binary(
    name = "generate_cpp",
    args = ["--stderrthreshold=1"],  # Warnings and errors to stderr.
    deps = [":generate_cpp_main"],
)

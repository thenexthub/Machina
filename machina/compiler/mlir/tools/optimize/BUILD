load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//machina:machina.default.bzl", "get_compatible_with_portable")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "quantization_utils",
    srcs = ["quantization_utils.cc"],
    hdrs = ["quantization_utils.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//machina/compiler/mlir/tools:safe_cast",
        "//machina/core/framework:tensor_shape",
        "//machina/core/platform:logging",
        "//machina/core/platform:macros",
        "@com_google_absl//absl/types:span",
        "@local_xla//xla/tsl/platform:status",
    ],
)

load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//machina/core/runtime_fallback:internal"],
    licenses = ["notice"],
)

package_group(
    name = "internal",
    packages = [
        "//machina/core/runtime_fallback/conversion/...",
    ],
)

cc_library(
    name = "conversion_alwayslink",
    srcs = [
        "static_registration.cc",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":conversion_lib",
        "@tf_runtime//:tensor",
    ],
    alwayslink = 1,
)

cc_library(
    name = "conversion_lib",
    srcs = ["conversion.cc"],
    hdrs = ["conversion.h"],
    deps = [
        "//machina/core:framework",
        "//machina/core:protos_all_cc",
        "//machina/core/common_runtime/eager:tensor_handle",
        "//machina/core/framework:tensor",
        "//machina/core/platform:status",
        "//machina/core/runtime_fallback/kernel:kernel_fallback_tensor",
        "//machina/core/runtime_fallback/kernel:tensor_util",
        "//machina/core/runtime_fallback/runtime:kernel_utils",
        "//machina/core/runtime_fallback/runtime:runtime_fallback_tensor",
        "@com_google_absl//absl/status",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:tensor",
    ],
)

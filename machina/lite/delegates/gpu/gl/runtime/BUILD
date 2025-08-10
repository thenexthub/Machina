load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//machina/lite/delegates/gpu/gl:__subpackages__"],
    licenses = ["notice"],
)

cc_library(
    name = "shared_buffer",
    hdrs = ["shared_buffer.h"],
    deps = [
        "//machina/lite/delegates/gpu/common:status",
        "//machina/lite/delegates/gpu/common:util",
        "//machina/lite/delegates/gpu/gl:gl_buffer",
        "//machina/lite/delegates/gpu/gl:object",
        "//machina/lite/delegates/gpu/gl:portable",
    ],
)

load(
    "//machina/core/platform:rules_cc.bzl",
    "cc_library",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = [
        "//machina:__subpackages__",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "png_io",
    srcs = ["png_io.cc"],
    hdrs = ["png_io.h"],
    features = ["-layering_check"],
    visibility = ["//visibility:public"],
    deps = [
        "//machina/core/platform:byte_order",
        "//machina/core/platform:logging",
        "//machina/core/platform:png",
        "//machina/core/platform:stringpiece",
        "//machina/core/platform:types",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings:string_view",
        "@png",
        "@zlib",
    ],
)

alias(
    name = "testdata",
    actual = "//machina/core/lib/png/testdata:png_testdata",
    visibility = ["//machina/core:__pkg__"],
)

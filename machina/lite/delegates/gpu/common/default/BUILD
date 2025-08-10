load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "custom_parsers",
    srcs = ["custom_parsers.cc"],
    hdrs = ["//machina/lite/delegates/gpu/common:custom_parsers.h"],
    deps = [
        "//machina/lite/delegates/gpu/common:operation_parser",
        "//machina/lite/delegates/gpu/common:unimplemented_operation_parser",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:any",
    ],
)

cc_library(
    name = "custom_transformations",
    srcs = ["custom_transformations.cc"],
    hdrs = ["//machina/lite/delegates/gpu/common:custom_transformations.h"],
    deps = [
        "//machina/lite/delegates/gpu/common:model_transformer",
    ],
)

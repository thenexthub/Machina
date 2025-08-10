load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//machina:machina.bzl", "if_google")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    licenses = ["notice"],
)

cc_library(
    name = "mla_test_utils",
    testonly = True,
    hdrs = ["mla_test_utils.h"],
    visibility = ["//visibility:private"],
    deps = [
        "//machina/core/platform:errors",
        "//machina/core/platform:status",
        "@com_google_absl//absl/strings",
    ] + if_google([
        "//third_party/mira/mlarchive:mla",
        "//third_party/mira/mlarchive/google:google_env",
        "//machina/cc/experimental/tfa:saved_model_converter",
        "//machina/cc/experimental/tfa:test_utils",
    ]),
)

cc_library(
    name = "mla_utils",
    hdrs = ["mla_utils.h"],
    visibility = ["//visibility:private"],
    deps = [
        "//machina/core/platform:errors",
        "//machina/core/platform:statusor",
        "@com_google_absl//absl/strings",
    ] + if_google([
        "//third_party/mira/mlarchive:mla",
        "//third_party/mira/mlarchive:status_macro",
        "//machina/cc/experimental/tfa:saved_model_converter",
        "//third_party/mira/mlarchive/google:google_env",
        "//machina/core/platform:path",
        "//machina/core/platform:status",
    ]),
)

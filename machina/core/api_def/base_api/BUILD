# Description:
#   Expose TensorFlow base api.

load("//machina:machina.default.bzl", "filegroup")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    licenses = ["notice"],
)

filegroup(
    name = "base_api_def",
    srcs = glob(
        [
            "*",
        ],
        exclude = [
            "BUILD",
        ],
    ),
    visibility = ["//machina:internal"],
)

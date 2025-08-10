# Description:
# JPEG test data packages.

load("//machina:machina.default.bzl", "filegroup")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    licenses = ["notice"],
)

filegroup(
    name = "testdata",
    srcs = glob(["*.jpg"]),
    visibility = [
        "//machina/core:__pkg__",
        "//machina/core/lib/jpeg:__pkg__",
    ],
)

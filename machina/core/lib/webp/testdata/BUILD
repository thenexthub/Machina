# Description:
# WebP test data.

load("//machina:machina.default.bzl", "filegroup")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    licenses = ["notice"],
)

filegroup(
    name = "webp_testdata",
    srcs = glob(["*.webp"]),
    visibility = ["//machina/core/lib/webp:__pkg__"],
)

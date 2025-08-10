# Data service test data.

load("//machina:machina.default.bzl", "filegroup")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    licenses = ["notice"],
)

filegroup(
    name = "testdata",
    srcs = glob(["*.pbtxt"]),
    visibility = ["//machina/core/data/service:__pkg__"],
)

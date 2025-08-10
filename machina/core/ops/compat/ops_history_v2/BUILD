# Description:
# Test for keeping the history of OpDefs for every major version of TensorFlow,
# to validate that we don't make backwards-incompatible changes in particular
# for v2.

load("//machina:machina.default.bzl", "filegroup")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    licenses = ["notice"],
)

filegroup(
    name = "ops_history_v2_srcs",
    srcs = glob([
        "*.pbtxt",
    ]),
    visibility = ["//machina/core/ops/compat:__pkg__"],
)

# User ops.

load(
    "//machina:machina.bzl",
    "tf_copts",
)
load(
    "//machina/core/platform:rules_cc.bzl",
    "cc_library",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = [
        "//machina/core:__pkg__",
    ],
    licenses = ["notice"],
)

# One target for all user ops.
cc_library(
    name = "user_ops_op_lib",
    srcs = glob(["*.cc"]),
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//machina/core:__pkg__"],
    deps = [
        "//machina/core:framework",
        "//machina/core:portable_gif_internal",
    ],
    alwayslink = 1,
)

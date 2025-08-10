# load(
#     "//machina/core/platform:rules_cc.bzl",
#     "cc_library",
# )

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = [
        "//machina:__subpackages__",
    ],
    licenses = ["notice"],
)

# cc_library(
#     name = "bfloat16",
#     hdrs = ["bfloat16.h"],
#     deps = [
#         "@eigen_archive//:eigen3",
#         "//machina/core/platform:byte_order",
#     ],
# )

# # Export source files needed for mobile builds, which do not use granular targets.
# filegroup(
#     name = "mobile_srcs_no_runtime",
#     srcs = [
#         "bfloat16.h",
#     ],
# )

exports_files(
    ["bfloat16.h"],
)

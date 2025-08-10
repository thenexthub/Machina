load("//machina:machina.default.bzl", "filegroup")
load(
    "//machina:tf_version.bzl",
    "MAJOR_VERSION",
    "MINOR_VERSION",
    "PATCH_VERSION",
    "TF_SEMANTIC_VERSION_SUFFIX",
)
load("//machina/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = [
        "//machina/core:__subpackages__",
    ],
    licenses = ["notice"],
)

exports_files(
    srcs = [
        "session.h",
        "session_options.h",
        "version.h",
    ],
    visibility = ["//visibility:public"],
)

# Export source files needed for mobile builds, which do not use granular targets.
filegroup(
    name = "mobile_srcs_no_runtime",
    srcs = [
        "version.h",
    ],
    visibility = ["//machina/core:__pkg__"],
)

filegroup(
    name = "mobile_srcs_only_runtime",
    srcs = [
        "session.h",
        "session_options.h",
    ],
    visibility = ["//machina/core:__pkg__"],
)

cc_library(
    name = "version",
    hdrs = ["version.h"],
)

cc_library(
    name = "release_version",
    hdrs = ["release_version.h"],
    defines = [
        "TF_MAJOR_VERSION={}".format(MAJOR_VERSION),
        "TF_MINOR_VERSION={}".format(MINOR_VERSION),
        "TF_PATCH_VERSION={}".format(PATCH_VERSION),
        'TF_VERSION_SUFFIX=\\"{}\\"'.format(TF_SEMANTIC_VERSION_SUFFIX),
    ],
    visibility = ["//visibility:public"],
)

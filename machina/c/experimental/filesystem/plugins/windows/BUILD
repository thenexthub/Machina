# Experimental windows filesystem plugin.
load("//machina:machina.bzl", "get_win_copts", "tf_cc_shared_object")
load("//machina/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    licenses = ["notice"],
)

# Filesystem implementation for Windows environment
tf_cc_shared_object(
    name = "windows_filesystem.dll",
    framework_so = [],
    linkstatic = False,
    tags = [
        "manual",
        "nobuilder",
        "notap",
    ],
    visibility = ["//visibility:public"],
    deps = [":windows_filesystem_impl"],
)

# The real implementation of the filesystem.
cc_library(
    name = "windows_filesystem_impl",
    srcs = ["windows_filesystem.cc"],
    copts = get_win_copts(),
    tags = [
        "manual",
        "nobuilder",
        "notap",
    ],
    deps = ["//machina/c/experimental/filesystem:filesystem_interface"],
)

# Cloud test data files.

load("//machina:machina.default.bzl", "filegroup")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    licenses = ["notice"],
)

filegroup(
    name = "application_default_credentials",
    srcs = [
        "application_default_credentials.json",
    ],
    visibility = ["//machina/core/platform/cloud:__pkg__"],
)

filegroup(
    name = "service_account_credentials",
    srcs = [
        "service_account_credentials.json",
    ],
    visibility = ["//machina/core/platform/cloud:__pkg__"],
)

filegroup(
    name = "service_account_public_key",
    srcs = [
        "service_account_public_key.txt",
    ],
    visibility = ["//machina/core/platform/cloud:__pkg__"],
)

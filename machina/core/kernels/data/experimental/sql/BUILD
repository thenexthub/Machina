# Description:
#   SQL library.

load("//machina/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "sql",
    srcs = [
        "driver_manager.cc",
        "sqlite_query_connection.cc",
    ],
    hdrs = [
        "driver_manager.h",
        "query_connection.h",
        "sqlite_query_connection.h",
    ],
    deps = [
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core/lib/db:sqlite",
    ],
)

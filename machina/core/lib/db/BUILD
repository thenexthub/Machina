# Description:
#   Libraries for storing tensors in SQL databases.

load("//machina:machina.bzl", "tf_cc_test", "tf_copts")
load(
    "//machina/core/platform:rules_cc.bzl",
    "cc_library",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//machina:internal"],
    licenses = ["notice"],
)

cc_library(
    name = "sqlite",
    srcs = ["sqlite.cc"],
    hdrs = ["sqlite.h"],
    copts = tf_copts(),
    deps = [
        ":snapfn",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@org_sqlite",
    ],
)

cc_library(
    name = "snapfn",
    srcs = ["snapfn.cc"],
    copts = tf_copts() + ["-DSQLITE_OMIT_LOAD_EXTENSION"],
    linkstatic = 1,
    deps = [
        "@org_sqlite",
        "@snappy",
    ],
    alwayslink = 1,
)

tf_cc_test(
    name = "sqlite_test",
    size = "small",
    srcs = ["sqlite_test.cc"],
    deps = [
        ":sqlite",
        "//machina/core:lib",
        "//machina/core:test",
        "//machina/core:test_main",
    ],
)

# Description:
# For keeping the history of OpDefs for every major version of TensorFlow,
# to validate that we don't make backwards-incompatible changes.

load(
    "//machina:machina.bzl",
    "tf_cc_binary",
    "tf_cc_test",
)
load("//machina/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//visibility:private"],
    licenses = ["notice"],
)

cc_library(
    name = "op_compatibility_lib",
    srcs = ["op_compatibility_lib.cc"],
    hdrs = ["op_compatibility_lib.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:ops",
        "//machina/core:protos_all_cc",
        "//machina/core/ops:debug_ops_op_lib",
    ],
)

tf_cc_test(
    name = "backwards_compatibility_test",
    size = "small",
    srcs = [
        "backwards_compatibility_test.cc",
    ],
    data = [
        "//machina/core/ops:ops.pbtxt",
        "//machina/core/ops/compat/ops_history_v1:ops_history_v1_srcs",
        "//machina/core/ops/compat/ops_history_v2:ops_history_v2_srcs",
    ] + glob([
        "ops_history.v*.pbtxt",
    ]),
    tags = [
        "no_windows",  # TODO(b/150030420): Issues with the path separator
    ],
    deps = [
        ":op_compatibility_lib",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core/public:release_version",
    ],
)

tf_cc_binary(
    name = "update_ops",
    srcs = ["update_ops_main.cc"],
    deps = [
        ":op_compatibility_lib",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core/platform:path",
        "//machina/core/public:release_version",
    ],
)

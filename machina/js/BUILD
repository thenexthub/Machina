# Description:
# JavaScript/TypeScript code generation for TensorFlow.js

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load(
    "//machina:machina.bzl",
    "tf_cc_test",
)

visibility = [
    "//machina:internal",
]

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = visibility,
    licenses = ["notice"],
)

cc_library(
    name = "ts_op_gen",
    srcs = [
        "ops/ts_op_gen.cc",
    ],
    hdrs = [
        "ops/ts_op_gen.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core:op_gen_lib",
        "//machina/core:protos_all_cc",
    ],
)

tf_cc_test(
    name = "ts_op_gen_test",
    srcs = [
        "ops/ts_op_gen.cc",
        "ops/ts_op_gen.h",
        "ops/ts_op_gen_test.cc",
    ],
    deps = [
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core:op_gen_lib",
        "//machina/core:protos_all_cc",
        "//machina/core:test",
        "//machina/core:test_main",
    ],
)

load("//machina:machina.bzl", "tf_cc_test")
load("//machina/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    licenses = ["notice"],
)

cc_library(
    name = "utils",
    srcs = [
        "utils.cc",
    ],
    hdrs = [
        "utils.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "@com_google_absl//absl/status",
    ],
)

tf_cc_test(
    name = "utils_test",
    srcs = [
        "utils_test.cc",
    ],
    deps = [
        ":utils",
        "//machina/core:lib",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core/framework:graph_proto_cc",
        "//machina/core/protobuf:for_core_protos_cc",
        "@com_google_absl//absl/status",
        "@local_tsl//tsl/platform:status",
    ],
)

cc_library(
    name = "input_yielder",
    hdrs = [
        "input_yielder.h",
    ],
    visibility = ["//visibility:public"],
    deps = [],
)

cc_library(
    name = "trivial_test_graph_input_yielder",
    srcs = ["trivial_test_graph_input_yielder.cc"],
    hdrs = [
        "trivial_test_graph_input_yielder.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":input_yielder",
        "//machina/cc:cc_ops",
        "//machina/cc:ops",
        "//machina/cc:scope",
        "//machina/core:all_kernels",
        "//machina/core:framework",
        "//machina/core:functional_ops_op_lib",
        "//machina/core:math_ops_op_lib",
        "//machina/core:no_op_op_lib",
        "//machina/core:portable_gif_internal",
        "//machina/core:protos_all_cc",
        "//machina/core:sendrecv_ops_op_lib",
        "//machina/core/grappler:grappler_item",
        "//machina/core/kernels:aggregate_ops",
        "//machina/core/platform:strcat",
        "@local_tsl//tsl/platform:status",
    ],
)

cc_library(
    name = "file_input_yielder",
    srcs = ["file_input_yielder.cc"],
    hdrs = [
        "file_input_yielder.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":input_yielder",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core/grappler:grappler_item",
        "//machina/core/grappler:grappler_item_builder",
        "//machina/core/grappler:utils",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
    ],
)

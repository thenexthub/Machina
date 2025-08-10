# Description:
# Graph C API.

load(
    "//machina:machina.bzl",
    "tf_cc_test",
)
load("//machina/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    licenses = ["notice"],
)

cc_library(
    name = "grappler_hdrs",
    hdrs = ["grappler.h"],
    visibility = ["//machina:internal"],
    deps = [
        "//machina/c:c_api",
        "//machina/c:c_api_macros",
        "//machina/c:tf_status_headers",
    ],
)

cc_library(
    name = "grappler",
    srcs = ["grappler.cc"],
    hdrs = [
        "grappler.h",
        "grappler_internal.h",
    ],
    visibility = ["//machina:internal"],
    deps = [
        "//machina/c:c_api_internal",
        "//machina/c:c_api_macros",
        "//machina/c:tf_buffer_internal",
        "//machina/c:tf_status",
        "//machina/c:tf_status_helper",
        "//machina/core:framework",
        "//machina/core:protos_all_cc",
        "//machina/core/grappler:grappler_item",
        "//machina/core/grappler/costs:graph_properties",
        "//machina/core/grappler/optimizers:custom_graph_optimizer",
        "//machina/core/grappler/optimizers:custom_graph_optimizer_registry",
        "//machina/core/platform:logging",
        "//machina/core/platform:status",
        "@com_google_absl//absl/status",
        "@local_xla//xla/tsl/platform:env",
        "@local_xla//xla/tsl/platform:errors",
    ],
)

tf_cc_test(
    name = "grappler_test",
    srcs = ["grappler_test.cc"],
    deps = [
        ":grappler",
        "//machina/c:tf_buffer_internal",
        "//machina/c:tf_status_headers",
        "//machina/core:framework",
        "//machina/core:portable_gif_internal",
        "//machina/core:protos_all_cc",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core/grappler:grappler_item",
        "//machina/core/grappler/clusters:single_machine",
        "//machina/core/grappler/costs:op_performance_data_cc",
        "//machina/core/grappler/inputs:trivial_test_graph_input_yielder",
        "//machina/core/grappler/optimizers:custom_graph_optimizer_registry",
        "//machina/core/platform:status",
        "//machina/core/protobuf:for_core_protos_cc",
        "@com_google_absl//absl/log:check",
        "@local_xla//xla/tsl/protobuf:error_codes_proto_impl_cc",
    ],
)

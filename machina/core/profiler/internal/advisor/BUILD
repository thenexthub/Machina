load("//machina:machina.bzl", "tf_cc_test")
load("//machina/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//machina:internal"],
    licenses = ["notice"],
)

cc_library(
    name = "checker",
    hdrs = ["checker.h"],
    deps = [
        "//machina/core:lib",
        "//machina/core/profiler:protos_all_cc",
        "//machina/core/profiler/internal:tfprof_stats",
    ],
)

cc_library(
    name = "internal_checker_runner_dummy",
    srcs = ["internal_checker_runner_dummy.cc"],
    hdrs = ["internal_checker_runner.h"],
    deps = [
        "//machina/core/profiler:protos_all_cc",
        "//machina/core/profiler/internal:tfprof_utils",
    ],
)

cc_library(
    name = "accelerator_utilization_checker",
    hdrs = ["accelerator_utilization_checker.h"],
    deps = [
        ":checker",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library(
    name = "operation_checker",
    hdrs = ["operation_checker.h"],
    deps = [
        ":checker",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library(
    name = "expensive_operation_checker",
    hdrs = ["expensive_operation_checker.h"],
    deps = [
        ":checker",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library(
    name = "tfprof_advisor",
    hdrs = ["tfprof_advisor.h"],
    deps = [
        ":accelerator_utilization_checker",
        ":checker",
        ":expensive_operation_checker",
        ":internal_checker_runner_dummy",
        ":operation_checker",
        "//machina/core/profiler:protos_all_cc",
        "@com_google_absl//absl/strings:str_format",
    ],
)

tf_cc_test(
    name = "tfprof_advisor_test",
    srcs = ["tfprof_advisor_test.cc"],
    deps = [
        ":checker",
        ":tfprof_advisor",
        "//machina/core:lib",
        "//machina/core:test",
        "//machina/core/framework:graph_proto_cc",
        "//machina/core/framework:node_def_proto_cc",
        "//machina/core/framework:step_stats_proto_cc",
        "//machina/core/profiler:protos_all_cc",
        "//machina/core/profiler/internal:tfprof_node",
        "//machina/core/profiler/internal:tfprof_stats",
        "//machina/core/profiler/internal:tfprof_tf_testlib",
        "@com_google_absl//absl/strings",
    ],
)

# tf.data service client library.

load("//machina:machina.bzl", "tf_cc_test")
load("//machina:machina.default.bzl", "tf_grpc_cc_dependencies")
load("//machina/core/platform:build_config.bzl", "tf_protos_profiler_service")
load("//machina/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = [
        "//machina:internal",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "common",
    hdrs = [
        "common.h",
    ],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        "//machina/core:protos_all_cc",
        "//machina/core/data/service:common_proto_cc",
        "@com_google_absl//absl/time",
    ],
)

cc_library(
    name = "data_service_client",
    srcs = ["data_service_client.cc"],
    hdrs = ["data_service_client.h"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        ":common",
        ":validate_utils",
        "//machina/core:framework",
        "//machina/core:framework_internal",
        "//machina/core:lib",
        "//machina/core/data:utils",
        "//machina/core/data/service:common",
        "//machina/core/data/service:common_proto_cc",
        "//machina/core/data/service:dispatcher_client",
        "//machina/core/data/service:dispatcher_proto_cc",
        "//machina/core/data/service:grpc_util",
        "//machina/core/data/service:worker_client",
        "//machina/core/data/service:worker_impl",
        "//machina/core/data/service:worker_proto_cc",
        "//machina/core/platform:errors",
        "//machina/core/platform:status",
        "//machina/core/platform:statusor",
        "//machina/core/platform:thread_annotations",
        "//machina/core/profiler/lib:traceme",
        "//machina/core/profiler/lib:traceme_encode",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@local_tsl//tsl/platform:retrying_utils",
        "@local_xla//xla/tsl/protobuf:error_codes_proto_impl_cc",
    ],
)

tf_cc_test(
    name = "data_service_client_test",
    srcs = ["data_service_client_test.cc"],
    # copybara:uncomment extra_copts = ["-Wthread-safety-analysis"],
    deps = [
        ":common",
        ":data_service_client",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core/data/service:common",
        "//machina/core/data/service:common_proto_cc",
        "//machina/core/data/service:test_cluster",
        "//machina/core/data/service:test_util",
        "//machina/core/platform:status",
        "//machina/core/platform:status_matchers",
        "//machina/core/platform:statusor",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest",
        "@local_xla//xla/tsl/protobuf:error_codes_proto_impl_cc",
    ] + tf_grpc_cc_dependencies() + tf_protos_profiler_service(),
)

cc_library(
    name = "utils",
    srcs = ["utils.cc"],
    hdrs = ["utils.h"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core/data/service:dispatcher_client",
        "//machina/core/data/service:dispatcher_proto_cc",
        "//machina/core/data/service:grpc_util",
        "//machina/core/platform:env_time",
        "//machina/core/platform:statusor",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@local_tsl//tsl/platform:errors",
        "@local_xla//xla/tsl/protobuf:protos_all_cc",
    ],
)

tf_cc_test(
    name = "utils_test",
    srcs = ["utils_test.cc"],
    # copybara:uncomment extra_copts = ["-Wthread-safety-analysis"],
    deps = [
        ":utils",
        "//machina/core:framework",
        "//machina/core:protos_all_cc",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core/data/service:dispatcher_client",
        "//machina/core/data/service:test_cluster",
        "//machina/core/data/service:test_util",
        "@com_google_googletest//:gtest",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:status_matchers",
        "@local_xla//xla/tsl/lib/core:status_test_util",
        "@local_xla//xla/tsl/protobuf:protos_all_cc",
    ] + tf_grpc_cc_dependencies() + tf_protos_profiler_service(),
)

cc_library(
    name = "validate_utils",
    srcs = ["validate_utils.cc"],
    hdrs = ["validate_utils.h"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        ":common",
        "//machina/core:framework",
        "//machina/core:protos_all_cc",
        "//machina/core/data/service:common",
        "//machina/core/data/service:common_proto_cc",
        "//machina/core/data/service:worker_impl",
        "//machina/core/platform:errors",
        "//machina/core/platform:status",
        "@com_google_absl//absl/status",
    ],
)

tf_cc_test(
    name = "validate_utils_test",
    srcs = ["validate_utils_test.cc"],
    # copybara:uncomment extra_copts = ["-Wthread-safety-analysis"],
    deps = [
        ":common",
        ":validate_utils",
        "//machina/core:framework",
        "//machina/core:protos_all_cc",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core/data/service:common_proto_cc",
        "//machina/core/data/service:worker_impl",
        "//machina/core/platform:status",
        "//machina/core/platform:status_matchers",
        "@com_google_googletest//:gtest_main",
        "@local_xla//xla/tsl/protobuf:error_codes_proto_impl_cc",
    ],
)

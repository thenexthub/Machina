load("//machina:machina.bzl", "tf_cc_shared_object", "tf_cc_test")
load("//machina/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    licenses = ["notice"],
)

cc_library(
    name = "plugin_c_api_hdrs",
    hdrs = ["plugin_c_api.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//machina/c:c_api_headers",
        "//machina/c:c_api_macros_hdrs",
        "//machina/c:tf_status_headers",
        "//machina/c:tf_tensor_hdrs",
        "@local_xla//xla/c:c_api_decl",
        "@local_xla//xla/pjrt/c:pjrt_c_api_hdrs",
        "@local_xla//xla/stream_executor/tpu:c_api_decl",
    ],
)

cc_library(
    name = "example_plugin",
    testonly = 1,
    srcs = ["example_plugin.cc"],
    hdrs = ["example_plugin.h"],
    deps = [
        ":plugin_c_api_hdrs",
        "//machina/core/platform:logging",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@local_tsl//tsl/platform:env",
        "@tf_runtime//:hostcontext_alwayslink",
    ],
)

tf_cc_test(
    name = "plugin_c_api_test",
    srcs = ["plugin_c_api_test.cc"],
    deps = [
        ":example_plugin",
        ":plugin_c_api_hdrs",
        "//machina/c:c_api",
        "//machina/c:tf_status_headers",
        "//machina/core/platform:status",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
        "@tf_runtime//:hostcontext_alwayslink",
    ],
)

tf_cc_shared_object(
    name = "test_next_pluggable_device_plugin.so",
    srcs = ["test_next_pluggable_device_plugin.cc"],
    visibility = ["//machina/c:__subpackages__"],
    deps = [
        ":plugin_c_api_hdrs",
        "@local_xla//xla/pjrt/c:pjrt_c_api_hdrs",
    ],
)

cc_library(
    name = "test_next_pluggable_device_plugin",
    srcs = ["test_next_pluggable_device_plugin.cc"],
    visibility = ["//machina/core:__subpackages__"],
    deps = [
        ":plugin_c_api_hdrs",
        "@local_xla//xla/pjrt/c:pjrt_c_api_hdrs",
    ],
)

cc_library(
    name = "outside_compilation_params",
    hdrs = ["outside_compilation_params.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":tf_rendezvous_c_api",
        "@local_xla//xla/stream_executor/tpu:c_api_decl",
    ],
)

cc_library(
    name = "tf_rendezvous_c_api",
    hdrs = ["tf_rendezvous_c_api.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//machina/c:c_api_macros_hdrs",
        "//machina/c:tf_status_headers",
        "//machina/c:tf_tensor_hdrs",
    ],
)

cc_library(
    name = "tf_rendezvous_c_api_defn",
    hdrs = ["tf_rendezvous_c_api_defn.h"],
    deps = [
        ":tf_rendezvous_c_api",
        "//machina/core:framework",
    ],
)

cc_library(
    name = "tf_rendezvous_c_api_helper",
    srcs = ["tf_rendezvous_c_api_helper.cc"],
    hdrs = ["tf_rendezvous_c_api_helper.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":outside_compilation_params",
        ":tf_device_context_c_api",
        ":tf_device_context_c_api_helper",
        ":tf_rendezvous_c_api",
        ":tf_rendezvous_c_api_defn",
        ":tf_tensor_utils",
        "//machina/c:tf_status_headers",
        "//machina/c:tf_status_helper",
        "//machina/c:tf_tensor_hdrs",
        "//machina/core:framework",
        "//machina/core:framework_internal",
        "//machina/core/platform:status",
        "//machina/core/platform:statusor",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:status",
        "@local_xla//xla/stream_executor/tpu:c_api_decl",
    ],
)

cc_library(
    name = "tf_rendezvous_c_api_internal",
    srcs = ["tf_rendezvous_c_api_internal.cc"],
    hdrs = ["tf_rendezvous_c_api_internal.h"],
    visibility = ["//machina:__subpackages__"],
    deps = [
        ":tf_device_context_c_api",
        ":tf_device_context_c_api_internal",
        ":tf_rendezvous_c_api",
        "//machina/c:tf_status",
        "//machina/c:tf_status_helper",
        "//machina/c:tf_tensor_hdrs",
        "//machina/core:framework",
        "//machina/core:framework_internal",
        "//machina/core/platform:status",
        "//machina/core/platform:statusor",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:status",
    ],
)

tf_cc_test(
    name = "tf_rendezvous_c_api_test",
    srcs = ["tf_rendezvous_c_api_test.cc"],
    tags = [
        "no_mac",
        "no_windows",
    ],
    deps = [
        ":tf_rendezvous_c_api",
        ":tf_rendezvous_c_api_helper",
        ":tf_rendezvous_c_api_internal",
        "//machina/c:tf_status_headers",
        "//machina/c:tf_status_helper",
        "//machina/c:tf_tensor",
        "//machina/core:core_cpu_base",
        "//machina/core:core_cpu_lib",
        "//machina/core:framework",
        "//machina/core:framework_internal",
        "//machina/core:lib",
        "//machina/core/framework:tensor_testutil",
        "//machina/core/framework:types_proto_cc",
        "//machina/core/platform:refcount",
        "//machina/core/platform:status",
        "//machina/core/platform:stringpiece",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:status",
        "@local_xla//xla/tsl/lib/core:status_test_util",
    ],
)

cc_library(
    name = "tf_device_context_c_api",
    hdrs = ["tf_device_context_c_api.h"],
)

cc_library(
    name = "tf_device_context_c_api_helper",
    srcs = ["tf_device_context_c_api_helper.cc"],
    hdrs = ["tf_device_context_c_api_helper.h"],
    deps = [
        ":tf_device_context_c_api",
        ":tf_tensor_utils",
        "//machina/c:tf_status_headers",
        "//machina/c:tf_status_helper",
        "//machina/core:framework",
        "//machina/core:framework_internal",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:status",
    ],
)

cc_library(
    name = "tf_device_context_c_api_internal",
    srcs = ["tf_device_context_c_api_internal.cc"],
    hdrs = ["tf_device_context_c_api_internal.h"],
    deps = [
        ":tf_device_context_c_api",
        "//machina/c:tf_status_headers",
        "//machina/c:tf_status_helper",
        "//machina/c:tf_tensor",
        "//machina/core:framework",
        "//machina/core:framework_internal",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:status",
    ],
)

cc_library(
    name = "tf_tensor_utils",
    srcs = ["tf_tensor_utils.cc"],
    hdrs = ["tf_tensor_utils.h"],
    deps = [
        ":tf_device_context_c_api",
        "//machina/c:tf_datatype_hdrs",
        "//machina/c:tf_tensor_hdrs",
        "//machina/core:framework",
        "//machina/core:framework_internal",
        "//machina/core/framework:types_proto_cc",
        "@local_xla//xla/tsl/framework:allocator",
    ],
)

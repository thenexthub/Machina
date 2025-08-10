load("//machina:machina.bzl", "tf_cc_test", "tf_copts")
load("//machina:machina.default.bzl", "filegroup")
load("//machina/core/platform:build_config_root.bzl", "if_static")
load("//machina/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = [
        "//machina:internal",
    ],
    licenses = ["notice"],
)

filegroup(
    name = "pluggable_device_runtime_headers",
    srcs = [
        "pluggable_device.h",
        "pluggable_device_bfc_allocator.h",
        "pluggable_device_context.h",
        "pluggable_device_factory.h",
        "pluggable_device_init.h",
        "pluggable_device_process_state.h",
        "pluggable_device_simple_allocator.h",
        "pluggable_device_util.h",
        "//machina/core/common_runtime/device:device_runtime_headers",
    ],
    visibility = ["//visibility:private"],
)

cc_library(
    name = "pluggable_device_runtime_impl",
    srcs = [
        "pluggable_device.cc",
        "pluggable_device_context.cc",
        "pluggable_device_factory.cc",
        "pluggable_device_process_state.cc",
        "pluggable_device_util.cc",
    ],
    hdrs = [":pluggable_device_runtime_headers"],
    copts = tf_copts(),
    deps = [
        ":pluggable_device_bfc_allocator",
        ":pluggable_device_init_impl",
        ":pluggable_device_simple_allocator",
        "//machina/c/experimental/stream_executor",
        "//machina/c/experimental/stream_executor:stream_executor_internal",
        "//machina/core:core_cpu",
        "//machina/core:core_cpu_lib",
        "//machina/core:framework",
        "//machina/core:framework_internal",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core:protos_all_cc",
        "//machina/core/common_runtime:device_id_utils",
        "//machina/core/common_runtime/device:device_event_mgr",
        "//machina/core/platform:stream_executor",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@local_xla//xla/stream_executor:device_memory",
        "@local_xla//xla/stream_executor:stream_executor_h",
        "@local_xla//xla/stream_executor/integrations:device_mem_allocator",
        "@local_xla//xla/stream_executor/integrations:stream_executor_allocator",
        "@local_xla//xla/tsl/framework:device_id_utils",
    ],
    alwayslink = 1,
)

cc_library(
    name = "pluggable_device_plugin_init",
    srcs = [
        "pluggable_device_plugin_init.cc",
    ],
    hdrs = [
        "pluggable_device_plugin_init.h",
        ":pluggable_device_runtime_headers",
    ],
    copts = tf_copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":pluggable_device_runtime_impl",
        "//machina/c/experimental/grappler",
        "//machina/c/experimental/pluggable_profiler",
        "//machina/c/experimental/pluggable_profiler:pluggable_profiler_internal",
        "//machina/c/experimental/stream_executor",
        "//machina/c/experimental/stream_executor:stream_executor_internal",
        "//machina/compiler/jit:flags_headers",
        "//machina/compiler/jit:pjrt_device_context",
        "//machina/compiler/jit:xla_device_no_jit_rewrite_registration",
        "//machina/core:core_cpu",
        "//machina/core:core_cpu_base",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core:protos_all_cc",
        "//machina/core/common_runtime:bfc_allocator",
        "//machina/core/common_runtime:dma_helper",
        "//machina/core/common_runtime:local_device",
        "//machina/core/common_runtime:process_state",
        "//machina/core/common_runtime:shared_counter",
        "//machina/core/common_runtime/next_pluggable_device:next_pluggable_device_api",
        "//machina/core/common_runtime/next_pluggable_device:next_pluggable_device_factory",
        "//machina/core/platform:stream_executor",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@local_xla//xla/pjrt:pjrt_api",
        "@local_xla//xla/pjrt/c:pjrt_c_api_hdrs",
        "@local_xla//xla/stream_executor:stream_executor_h",
        "@local_xla//xla/stream_executor/integrations:device_mem_allocator",
    ] + if_static([
        "//machina/core/common_runtime:copy_tensor",
    ]),
)

cc_library(
    name = "pluggable_device_runtime",
    hdrs = [":pluggable_device_runtime_headers"],
    linkstatic = 1,
    deps = [
        "//machina/core:core_cpu",
        "//machina/core:core_cpu_base",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core:protos_all_cc",
        "//machina/core/common_runtime:core_cpu",
        "//machina/core/common_runtime:dma_helper",
        "//machina/core/common_runtime:shared_counter",
        "//machina/core/lib/core:status",
        "//machina/core/platform:stream_executor",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@local_xla//xla/stream_executor:stream_executor_h",
        "@local_xla//xla/stream_executor/integrations:device_mem_allocator",
    ] + if_static([
        # Temporary workaround for duplicated symbols issues.
        ":pluggable_device_runtime_impl",
        "//machina/core/common_runtime:bfc_allocator",
        "//machina/core/common_runtime:local_device",
        "//machina/core/common_runtime:process_state",
    ]),
)

cc_library(
    name = "pluggable_device_bfc_allocator",
    srcs = [
        "pluggable_device_bfc_allocator.cc",
    ],
    hdrs = ["pluggable_device_bfc_allocator.h"],
    features = ["parse_headers"],
    visibility = ["//visibility:public"],
    deps = [
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core:protos_all_cc",
        "//machina/core/common_runtime:bfc_allocator",
        "//machina/core/common_runtime/device:device_mem_allocator",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/memory",
        "@local_xla//xla/tsl/framework:allocator",
        "@local_xla//xla/tsl/framework:bfc_allocator",
    ],
)

cc_library(
    name = "pluggable_device_simple_allocator",
    srcs = [
        "pluggable_device_simple_allocator.cc",
    ],
    hdrs = ["pluggable_device_simple_allocator.h"],
    features = ["parse_headers"],
    visibility = ["//visibility:public"],
    deps = [
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core:protos_all_cc",
        "//machina/core/framework:allocator",
        "@local_xla//xla/tsl/framework:allocator",
    ],
)

cc_library(
    name = "pluggable_device_init",
    hdrs = [
        "pluggable_device_init.h",
    ],
    deps = [
        "pluggable_device_init_impl",
        "//machina/core:framework",
        "//machina/core:framework_internal",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core/platform:stream_executor",
        "@com_google_absl//absl/status",
    ],
)

cc_library(
    name = "pluggable_device_init_impl",
    srcs = [
        "pluggable_device_init.cc",
    ],
    hdrs = [
        "pluggable_device_init.h",
    ],
    copts = tf_copts(),
    linkstatic = 1,
    deps = [
        "//machina/core:core_cpu_lib",
        "//machina/core:framework",
        "//machina/core:framework_internal",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core/platform:stream_executor",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@local_xla//xla/stream_executor:platform_manager",
    ],
    alwayslink = 1,
)

tf_cc_test(
    name = "pluggable_device_plugin_init_test",
    size = "small",
    srcs = ["pluggable_device_plugin_init_test.cc"],
    deps = [
        ":pluggable_device_plugin_init",
        "//machina/c:c_api",
        "//machina/c/experimental/grappler:grappler_hdrs",
        "//machina/c/experimental/pluggable_profiler:pluggable_profiler_hdrs",
        "//machina/c/experimental/stream_executor:stream_executor_hdrs",
        "//machina/c/experimental/stream_executor/test:test_pluggable_device",  # buildcleaner: keep
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core/common_runtime/next_pluggable_device/c:plugin_c_api_hdrs",
        "//machina/core/common_runtime/next_pluggable_device/c:test_next_pluggable_device_plugin",  # buildcleaner: keep
        "@com_google_googletest//:gtest_main",
        "@local_xla//xla/pjrt/c:pjrt_c_api_hdrs",
    ],
)

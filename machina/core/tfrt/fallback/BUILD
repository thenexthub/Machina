load(
    "//machina:machina.bzl",
    "if_android",
    "if_mobile",
    "if_not_mobile",
    "tf_cc_test",
    "tf_features_nolayering_check_if_ios",
)
load("//machina:machina.default.bzl", "tf_cuda_cc_test")
load("//machina/core/platform:build_config.bzl", "tf_proto_library")
load(
    "//machina/core/platform:build_config_root.bzl",
    "if_static",
    "tf_cuda_tests_tags",
)
load("//machina/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = [":friends"],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    packages = [
        # Authorized users go here.
        "//machina/compiler/mlir/...",
        "//machina/core/...",
        # copybara:uncomment "//learning/brain/experimental/tfrt/...",
        # copybara:uncomment "//learning/brain/tfrt/...",
        # copybara:uncomment "//learning/brain/mobile/lite/delegates/tfmrt/...",
        # copybara:uncomment "//learning/infra/mira/experimental/orbax_model/...",
    ],
)

cc_library(
    name = "fallback_state",
    srcs = ["fallback_state.cc"],
    hdrs = ["fallback_state.h"],
    deps = [
        "//machina/core:core_cpu_base",
        "//machina/core:framework",
        "//machina/core:framework_internal",
        "//machina/core:portable_gif_internal",
        "//machina/core:session_options",
        "//machina/core/common_runtime:core_cpu_internal",
        "//machina/core/common_runtime:device_set",
        "//machina/core/framework:device_attributes_proto_cc",
        "//machina/core/framework:function_proto_cc",
        "//machina/core/framework:graph_proto_cc",
        "//machina/core/platform:strcat",
        "//machina/core/tpu:virtual_device",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:refcount",
        "@local_xla//xla/tsl/platform:errors",
    ],
)

tf_cc_test(
    name = "fallback_state_test",
    srcs = ["fallback_state_test.cc"],
    deps = [
        ":fallback_state",
        "//machina/cc:cc_ops",
        "//machina/cc:const_op",
        "//machina/cc:ops",
        "//machina/cc:scope",
        "//machina/core:all_kernels",
        "//machina/core:core_cpu_base",
        "//machina/core:framework",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core/framework:function_proto_cc",
        "//machina/core/platform:status_matchers",
        "//machina/core/protobuf:error_codes_proto_impl_cc",
        "@com_google_absl//absl/base:nullability",
        "@com_google_googletest//:gtest",
        "@local_xla//xla/tsl/protobuf:error_codes_proto_impl_cc",
    ],
)

cc_library(
    name = "op_kernel_runner",
    srcs = ["op_kernel_runner.cc"],
    hdrs = ["op_kernel_runner.h"],
    features = tf_features_nolayering_check_if_ios() + if_android(["-layering_check"]),
    visibility = [
        # copybara:uncomment "//machina/core/runtime_fallback:internal",
        "//visibility:public",
    ],
    deps = [
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
    ] + if_mobile([
        "//machina/core:portable_machina_lib_lite",
    ]) + if_not_mobile([
        "//machina/core:framework",
        "//machina/core:core_cpu_base",
        "//machina/core/framework:node_def_proto_cc",
        "//machina/core/framework:op_def_proto_cc",
        "//machina/core/platform:errors",
        "//machina/core/platform:status",
    ]),
)

cc_library(
    name = "op_kernel_runner_cache",
    srcs = ["op_kernel_runner_cache.cc"],
    hdrs = ["op_kernel_runner_cache.h"],
    deps = [
        ":op_kernel_runner",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@tf_runtime//:hostcontext",
    ],
)

cc_library(
    name = "cost_recorder",
    srcs = ["cost_recorder.cc"],
    hdrs = ["cost_recorder.h"],
    deps = [
        ":op_cost_map_proto_cc",
        "//machina/core:lib",
        "//machina/core/platform:status",
        "//machina/core/util:env_var",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
    ],
)

cc_library(
    name = "device_with_custom_allocator",
    hdrs = ["device_with_custom_allocator.h"],
    deps = [
        "//machina/core:framework",
        "@local_xla//xla/tsl/framework:allocator",
    ],
)

tf_cc_test(
    name = "cost_recorder_test",
    srcs = ["cost_recorder_test.cc"],
    deps = [
        ":cost_recorder",
        ":op_cost_map_proto_cc",
        "//machina/core:lib",
        "@com_google_googletest//:gtest_main",
    ],
)

tf_cuda_cc_test(
    name = "op_kernel_runner_test",
    size = "small",
    srcs = ["op_kernel_runner_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":fallback_state",
        ":op_kernel_runner",
        ":op_kernel_runner_cache",
        "//machina/core:framework",
        "//machina/core:protos_all_cc",
        "//machina/core:session_options",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core:testlib",
    ] + if_static(
        [
            "//machina/core/common_runtime:function",
        ],
        [],
    ),
)

tf_proto_library(
    name = "op_cost_map_proto",
    srcs = ["op_cost_map.proto"],
)

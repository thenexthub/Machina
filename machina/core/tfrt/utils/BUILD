load("//machina:machina.bzl", "if_google", "if_oss", "tf_cc_shared_test", "tf_cc_test")
load("//machina/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = [":friends"],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    packages = [
        # copybara:uncomment "//learning/brain/experimental/tfrt/...",
        "//learning/brain/google/xla/kernels/...",
        # copybara:uncomment "//learning/brain/research/pjrt/...",
        # copybara:uncomment "//learning/brain/tfrt/...",
        # copybara:uncomment "//learning/pathways/serving/...",
        "//learning/serving/...",
        "//quality/webanswers/...",
        "//smartass/brain/inference/...",
        # copybara:uncomment "//smartass/brain/ops/...",
        "//machina/c/eager/...",
        "//machina/compiler/mlir/tfrt/...",
        "//machina/core/runtime_fallback/...",
        "//machina/core/tfrt/...",
        "//machina/python/...",
        # copybara:uncomment "//machina_serving/...",
        # copybara:uncomment "//third_party/tf_runtime_google/...",
    ],
)

cc_library(
    name = "utils",
    srcs = [
        "utils.cc",
    ],
    hdrs = [
        "dtype.def",
        "utils.h",
    ],
    deps = [
        ":error_util",
        "//machina/core:framework",
        "//machina/core/framework:types_proto_cc",
        "//machina/core/lib/gtl:array_slice",
        "//machina/core/platform:errors",
        "//machina/core/platform:profile_utils_cpu_utils",
        "//machina/core/platform:statusor",
        "//machina/core/platform:strcat",
        "//machina/core/tfrt/runtime",
        "//machina/core/tpu:virtual_device",
        "@com_google_absl//absl/status",
        "@local_xla//xla:status_macros",
        "@tf_runtime//:bef",
        "@tf_runtime//:befexecutor",
        "@tf_runtime//:core_runtime",
        "@tf_runtime//:dtype",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
    ],
)

tf_cc_shared_test(
    name = "utils_test",
    srcs = ["utils_test.cc"],
    tags = ["no_oss"],
    deps = [
        ":utils",
        "//machina/c:tf_tensor",
        "//machina/core:framework",
        "//machina/core/platform:errors",
        "//machina/core/platform:status",
        "//machina/core/platform:statusor",
        "@com_google_googletest//:gtest_main",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//cpp_tests:common",
    ],
)

cc_library(
    name = "tensor_util",
    srcs = ["tensor_util.cc"],
    hdrs = [
        "dtype.def",
        "tensor_util.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//machina/core/framework:tensor_shape",
        "//machina/core/framework:types_proto_cc",
        "//machina/core/platform:errors",
        "//machina/core/platform:statusor",
        "//machina/core/platform:tstring",
        "//machina/core/runtime_fallback/kernel:kernel_fallback_tensor",
        "//machina/core/runtime_fallback/util:tensor_util",
        "//machina/core/runtime_fallback/util:type_util",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings",
        "@eigen_archive//:eigen3",
        "@tf_runtime//:core_runtime",
        "@tf_runtime//:dtype",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
        "@tf_runtime//:tensor",
    ] + select({
        "//machina:android": [
            "//machina/core:portable_machina_lib_lite",  # TODO(annarev): exclude runtime srcs
        ],
        "//conditions:default": ["//machina/core/framework:tensor"],
    }),
)

tf_cc_test(
    name = "tensor_util_test",
    srcs = ["tensor_util_test.cc"],
    tags = ["no_oss"],
    deps = [
        ":tensor_util",
        "//machina/c:tf_tensor",
        "//machina/core:framework",
        "//machina/core/runtime_fallback/kernel:kernel_fallback_tensor",
        "@com_google_googletest//:gtest_main",
        "@toolchain-project//toolchain:Support",
        "@tf_runtime//:core_runtime",
        "@tf_runtime//:dtype",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
        "@tf_runtime//:tensor",
        "@tf_runtime//cpp_tests:common",
    ],
)

cc_library(
    name = "error_util",
    srcs = [
        "error_util.cc",
    ],
    hdrs = [
        "error_type.def",
        "error_util.h",
    ],
    deps = [
        "//machina/core/platform:status",
        "@com_google_absl//absl/status",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
    ],
)

tf_cc_test(
    name = "error_util_test",
    srcs = ["error_util_test.cc"],
    tags = ["no_oss"],
    deps = [
        ":error_util",
        "//machina/core/platform:status",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
        "@local_xla//xla/tsl/concurrency:async_value",
        "@tf_runtime//:support",
    ],
)

cc_library(
    name = "graph_partition",
    srcs = ["graph_partition.cc"],
    hdrs = ["graph_partition.h"],
    deps = [
        "//machina/core:framework",
        "//machina/core:graph",
        "//machina/core:lib",
        "//machina/core/common_runtime:device_set",
        "//machina/core/common_runtime:graph_constructor",
        "//machina/core/common_runtime:partitioning_utils",
        "//machina/core/grappler:utils",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
    ],
)

tf_cc_test(
    name = "graph_partition_test",
    srcs = ["graph_partition_test.cc"],
    deps = [
        ":graph_partition",
        "//machina/cc:cc_ops",
        "//machina/cc:sendrecv_ops",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core:testlib",
        "//machina/core/common_runtime:core_cpu",
        "//machina/core/common_runtime:inline_function_utils",
        "//machina/core/common_runtime:placer",
        "//machina/core/framework:graph_proto_cc",
        "//machina/core/grappler/utils:grappler_test",
    ],
)

cc_library(
    name = "tfrt_graph_execution_state",
    srcs = ["tfrt_graph_execution_state.cc"],
    hdrs = ["tfrt_graph_execution_state.h"],
    deps = [
        "//machina/compiler/mlir/machina:mlir_roundtrip_flags",
        "//machina/compiler/mlir/tf2xla/api/v1:mlir_bridge_config_v1_proto_cc",
        "//machina/compiler/tf2xla:functionalize_control_flow",
        "//machina/core:core_cpu_base",
        "//machina/core:framework",
        "//machina/core/common_runtime:core_cpu_internal",
        "//machina/core/framework:attr_value_proto_cc",
        "//machina/core/framework:function_proto_cc",
        "//machina/core/framework:graph_proto_cc",
        "//machina/core/framework:op_def_proto_cc",
        "//machina/core/framework:versions_proto_cc",
        "//machina/core/grappler:utils",
        "//machina/core/platform:errors",
        "//machina/core/platform:status",
        "//machina/core/platform:statusor",
        "//machina/core/protobuf:for_core_protos_cc",
        "//machina/core/tfrt/fallback:fallback_state",
        "//machina/core/tfrt/graph_executor:config",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@local_xla//xla/tsl/platform:errors",
    ],
)

tf_cc_test(
    name = "tfrt_graph_execution_state_test",
    srcs = ["tfrt_graph_execution_state_test.cc"],
    tags = if_oss([
        "manual",
        "no_oss",
    ]),  # b/169705709, no protobuf matchers in OSS.
    deps = [
        ":tfrt_graph_execution_state",
        "//machina/cc:array_ops",
        "//machina/cc:cc_ops",
        "//machina/cc:const_op",
        "//machina/cc:functional_ops",
        "//machina/cc:ops",
        "//machina/cc:scope",
        "//machina/cc:while_loop",
        "//machina/compiler/mlir:mlir_graph_optimization_pass_registration",
        "//machina/compiler/mlir/machina:attribute_utils",
        "//machina/compiler/mlir/machina:mlir_roundtrip_flags",
        "//machina/core:framework",
        "//machina/core:test",
        "//machina/core/framework:graph_proto_cc",
        "//machina/core/framework:tensor_testutil",
        "//machina/core/framework:types_proto_cc",
        "//machina/core/grappler/utils:grappler_test",
        "//machina/core/lib/monitoring:cell_reader",
        "//machina/core/platform:status",
        "//machina/core/protobuf:for_core_protos_cc",
        "//machina/core/tfrt/fallback:fallback_state",
        "//machina/core/tfrt/graph_executor:config",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_googletest//:gtest_main",
        "@local_xla//xla/tsl/platform:statusor",
    ],
)

cc_library(
    name = "fallback_tensor",
    srcs = ["fallback_tensor.cc"],
    hdrs = ["fallback_tensor.h"],
    deps = [
        "//machina/core/common_runtime:dma_helper",
        "//machina/core/framework:tensor",
        "@com_google_absl//absl/log:check",
        "@local_tsl//tsl/profiler/lib:traceme",
    ],
)

tf_cc_test(
    name = "fallback_tensor_test",
    srcs = ["fallback_tensor_test.cc"],
    tags = ["no_oss"],
    deps = [
        ":fallback_tensor",
        "//machina/core/common_runtime:dma_helper",
        "//machina/core/framework:tensor_shape",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "bridge_graph_analysis",
    hdrs = ["bridge_graph_analysis.h"],
    visibility = [
        "//machina/core/tfrt/graph_executor:__pkg__",
    ],
    deps = if_google([
        "//learning/brain/mlir/bridge:graph_analysis",
        "//machina/core/platform:enable_tf2_utils",
    ]) + [
        "//machina/core:core_cpu_base",
        "//machina/core/protobuf:for_core_protos_cc",
    ],
)

cc_library(
    name = "thread_pool",
    hdrs = ["thread_pool.h"],
    visibility = [
        ":friends",
        # copybara:uncomment "//smartass/brain/server:__pkg__",
    ],
    deps = [
        "//machina/core:lib",
        "//machina/core/platform:threadpool_interface",
    ],
)

cc_library(
    name = "device_variables_table",
    hdrs = ["device_variables_table.h"],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/synchronization",
        "@toolchain-project//toolchain:Support",
        "@tf_runtime//:hostcontext",
    ],
)

cc_library(
    name = "gpu_variables_table",
    hdrs = ["gpu_variables_table.h"],
    deps = [
        ":device_variables_table",
        ":fallback_tensor",
    ],
)

cc_library(
    name = "any_ptr",
    hdrs = ["any_ptr.h"],
)

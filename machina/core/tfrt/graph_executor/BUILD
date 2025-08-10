load("@rules_cc//cc:cc_library.bzl", "cc_library")

# Placeholder: load py_proto_library
load("//machina:machina.bzl", "if_google", "tf_cc_shared_object", "tf_cc_test")
load("//machina:machina.default.bzl", "get_compatible_with_portable")
load("//machina/core/platform:build_config.bzl", "tf_proto_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = [":friends"],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    packages = [
        # copybara:uncomment "//cloud/ai/platform/prediction/...",
        # copybara:uncomment "//learning/brain/experimental/tfrt/native_lowering/...",
        # copybara:uncomment "//learning/brain/tfrt/...",
        # copybara:uncomment "//learning/infra/mira/experimental/orbax_model/...",
        # copybara:uncomment "//learning/serving/servables/tfrt/...",
        # copybara:uncomment "//smartass/brain/inference/...",
        # copybara:uncomment "//machina/compiler/mlir/tfrt/...",
        "//machina/core/tfrt/...",
        "//machina/core/tfrt/graph_executor/python/...",
        # copybara:uncomment "//machina_serving/servables/machina/...",
        # copybara:uncomment "//machina_serving/servables/machina/google/...",
    ],
)

cc_library(
    name = "graph_execution_options",
    srcs = ["graph_execution_options.cc"],
    hdrs = ["graph_execution_options.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = ["//visibility:public"],
    deps = [
        ":config",
        "//machina/compiler/mlir/tfrt:tfrt_compile_options",
        "//machina/core:core_cpu",
        "//machina/core/framework:tensor",
        "//machina/core/protobuf:for_core_protos_cc",
        "//machina/core/tfrt/runtime:work_queue_interface",
        "//machina/core/tfrt/utils:bridge_graph_analysis",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:optional",
    ],
)

# For bazel python deps only
tf_cc_shared_object(
    name = "graph_execution_options.so",
    deps = [":graph_execution_options"],
)

cc_library(
    name = "executable_context",
    hdrs = ["executable_context.h"],
    deps = [
        "//machina/core/tfrt/mlrt/bytecode",
        "//machina/core/tfrt/mlrt/interpreter:context",
        "@tf_runtime//:bef",
        "@tf_runtime//:befexecutor",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
    ],
)

cc_library(
    name = "graph_executor",
    srcs = ["graph_executor.cc"],
    hdrs = ["graph_executor.h"],
    deps = [
        ":executable_context",
        ":export_mlir",
        ":graph_execution_options",
        ":sync_resource_state",
        "//machina/compiler/mlir/machina",
        "//machina/compiler/mlir/machina:error_util",
        "//machina/compiler/mlir/machina:import_model",
        "//machina/compiler/mlir/machina:mlir_roundtrip_flags",
        "//machina/compiler/mlir/tf2xla/api/v2:graph_to_tf_executor",
        "//machina/compiler/mlir/tfrt:backend_compiler",
        "//machina/compiler/mlir/tfrt:import_model",
        "//machina/compiler/mlir/tfrt:tfrt_compile_options",
        "//machina/compiler/mlir/tfrt:transforms/update_op_cost_in_tfrt_mlir",
        "//machina/compiler/mlir/tfrt/transforms/mlrt:import_model",
        "//machina/core:core_cpu_base",
        "//machina/core:framework",
        "//machina/core:framework_internal",
        "//machina/core:lib",
        "//machina/core/common_runtime:core_cpu_internal",
        "//machina/core/framework:tensor",
        "//machina/core/platform:errors",
        "//machina/core/platform:path",
        "//machina/core/profiler/lib:connected_traceme",
        "//machina/core/profiler/lib:traceme_encode",
        "//machina/core/protobuf:for_core_protos_cc",
        "//machina/core/public:version",
        "//machina/core/runtime_fallback/kernel:kernel_fallback_compat_request_state",
        "//machina/core/runtime_fallback/kernel:kernel_fallback_execute_compat",
        "//machina/core/runtime_fallback/kernel:kernel_fallback_utils",
        "//machina/core/tfrt/common:metrics",
        "//machina/core/tfrt/fallback:cost_recorder",
        "//machina/core/tfrt/fallback:fallback_state",
        "//machina/core/tfrt/fallback:op_kernel_runner",
        "//machina/core/tfrt/mlrt/bytecode",
        "//machina/core/tfrt/mlrt/bytecode:executable",
        "//machina/core/tfrt/mlrt/bytecode:function",
        "//machina/core/tfrt/mlrt/interpreter:context",
        "//machina/core/tfrt/mlrt/interpreter:execute",
        "//machina/core/tfrt/mlrt/interpreter:value",
        "//machina/core/tfrt/mlrt/kernel:context",
        "//machina/core/tfrt/runtime",
        "//machina/core/tfrt/runtime:step_id",
        "//machina/core/tfrt/runtime:stream",
        "//machina/core/tfrt/runtime:work_queue_interface",
        "//machina/core/tfrt/stubs:tfrt_native_lowering_stub",
        "//machina/core/tfrt/utils",
        "//machina/core/tfrt/utils:fallback_tensor",
        "//machina/core/tfrt/utils:tfrt_graph_execution_state",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:FuncExtensions",
        "@toolchain-project//mlir:IR",
        "@local_tsl//tsl/platform:refcount",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/profiler/lib:traceme",
        "@local_xla//xla/tsl/concurrency:async_value",
        "@local_xla//xla/tsl/concurrency:ref_count",
        "@tf_runtime//:basic_kernels_alwayslink",
        "@tf_runtime//:bef",
        "@tf_runtime//:befexecutor",
        "@tf_runtime//:core_runtime",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:mlirtobef",
        "@tf_runtime//:support",
    ] + if_google(
        [
            "//learning/brain/experimental/tfrt/native_lowering/stubs:tfrt_native_lowering_impl",
        ],
    ),
)

tf_cc_test(
    name = "graph_executor_test",
    srcs = ["graph_executor_test.cc"],
    tags = ["no_oss"],
    deps = [
        ":config",
        ":graph_execution_options",
        ":graph_executor",
        "//machina/cc:array_ops",
        "//machina/cc:cc_ops",
        "//machina/cc:const_op",
        "//machina/cc:ops",
        "//machina/cc:scope",
        "//machina/core:core_cpu_base",
        "//machina/core:framework",
        "//machina/core:framework_lite",
        "//machina/core:framework_types_hdr",
        "//machina/core:test",
        "//machina/core/framework:common_shape_fns",
        "//machina/core/framework:graph_proto_cc",
        "//machina/core/framework:op",
        "//machina/core/framework:tensor",
        "//machina/core/framework:types_proto_cc",
        "//machina/core/grappler/utils:grappler_test",
        "//machina/core/platform:status",
        "//machina/core/platform:statusor",
        "//machina/core/protobuf:for_core_protos_cc",
        "//machina/core/tfrt/fallback:fallback_state",
        "//machina/core/tfrt/mlrt/interpreter:context",
        "//machina/core/tfrt/mlrt/interpreter:value",
        "//machina/core/tfrt/mlrt/kernel",
        "//machina/core/tfrt/saved_model:saved_model_testutil",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:statusor",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:tensor",
        "@tf_runtime//cpp_tests:common",
    ] + if_google(
        [
            "//learning/brain/experimental/tfrt/native_lowering/kernels:kernels_alwayslink",
        ],
    ),
)

cc_library(
    name = "config",
    srcs = ["config.cc"],
    hdrs = ["config.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = ["//visibility:public"],
    deps = [
        ":config_proto_cc",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_protobuf//:protobuf",
    ],
)

tf_proto_library(
    name = "config_proto",
    srcs = ["config.proto"],
    visibility = ["//visibility:public"],
)

# copybara:uncomment_begin(google-only)
# py_proto_library(
#     name = "config_proto_py_pb2",
#     visibility = ["//visibility:public"],
#     deps = [":config_proto"],
# )
# copybara:uncomment_end

tf_proto_library(
    name = "test_config_proto",
    testonly = True,
    srcs = ["test_config.proto"],
    visibility = if_google(
        [":friends"],
        ["//visibility:public"],
    ),
)

tf_cc_test(
    name = "config_test",
    srcs = ["config_test.cc"],
    deps = [
        ":config",
        ":config_proto_cc",
        ":test_config_proto_cc",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:status_matchers",
        "@local_tsl//tsl/platform:statusor",
        "@local_xla//xla/tsl/lib/core:status_test_util",
    ],
)

cc_library(
    name = "sync_resource_state",
    hdrs = ["sync_resource_state.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//machina/core/tfrt/utils:any_ptr",
        "@tf_runtime//:tensor",
    ],
)

cc_library(
    name = "export_mlir",
    hdrs = ["export_mlir.h"],
    deps = ["@toolchain-project//mlir:IR"],
)

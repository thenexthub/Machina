load("//machina:machina.bzl", "tf_cc_test")
load(
    "//machina/core/platform:build_config.bzl",
    "tf_proto_library",
)
load("//machina/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = [
        "//machina/compiler/mlir/machina/transforms:__pkg__",
        "//machina/compiler/mlir/tf2xla/api/v1:__subpackages__",
        "//machina/compiler/mlir/tf2xla/api/v2:__subpackages__",
    ],
)

cc_library(
    name = "compilation_timer",
    hdrs = ["compilation_timer.h"],
    deps = [
        "//machina/core/platform:profile_utils_cpu_utils",
    ],
)

tf_cc_test(
    name = "compilation_timer_test",
    srcs = ["compilation_timer_test.cc"],
    deps = [
        ":compilation_timer",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "test_matchers",
    testonly = True,
    hdrs = ["test_matchers.h"],
    deps = [
        "//machina/compiler/mlir/tf2xla:compile_mlir_util",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest",
        "@local_xla//xla/tsl/platform:statusor",
    ],
)

tf_cc_test(
    name = "test_matchers_test",
    srcs = ["test_matchers_test.cc"],
    deps = [
        ":test_matchers",
        "//machina/compiler/mlir/tf2xla:compile_mlir_util",
        "//machina/compiler/tf2xla:xla_helpers",
        "//machina/core:lib",
        "//machina/core/lib/monitoring:cell_reader",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest_main",
        "@local_xla//xla/hlo/builder:xla_computation",
        "@local_xla//xla/service:hlo_proto_cc",
        "@local_xla//xla/tsl/platform:statusor",
    ],
)

cc_library(
    name = "mlir_pass_instrumentation",
    srcs = ["mlir_pass_instrumentation.cc"],
    hdrs = ["mlir_pass_instrumentation.h"],
    deps = [
        "//machina/core/platform:logging",
        "@com_google_absl//absl/log",
        "@toolchain-project//mlir:Pass",
    ],
)

tf_cc_test(
    name = "mlir_pass_instrumentation_test",
    srcs = ["mlir_pass_instrumentation_test.cc"],
    deps = [
        ":mlir_pass_instrumentation",
        "//machina/compiler/mlir/tf2xla/api/v1:compile_mlir_util_no_tf_dialect_passes",
        "//machina/core:test",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "legalize_tf_mlir",
    srcs = ["legalize_tf_mlir.cc"],
    hdrs = ["legalize_tf_mlir.h"],
    deps = [
        ":compilation_timer",
        "//machina/compiler/mlir/machina",
        "//machina/compiler/mlir/machina:serialize_mlir_module_utils",
        "//machina/compiler/mlir/machina/transforms:set_tpu_infeed_layout",
        "//machina/compiler/mlir/tf2xla:compile_mlir_util",
        "//machina/compiler/tf2xla:layout_util",
        "//machina/compiler/tf2xla:xla_compiler",
        "//machina/compiler/tf2xla:xla_helpers",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core/protobuf/tpu:compile_metadata_proto_cc",
        "//machina/core/tpu:tpu_compile",
        "//machina/core/tpu/kernels:tpu_compile_op_support",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Pass",
        "@local_tsl//tsl/platform:error_logging",
        "@local_xla//xla:shape_util",
        "@local_xla//xla/mlir_hlo:hlo_dialect_registration",
        "@local_xla//xla/tsl/platform:statusor",
        "@stablehlo//:register",
    ],
)

tf_cc_test(
    name = "legalize_tf_mlir_test",
    srcs = ["legalize_tf_mlir_test.cc"],
    deps = [
        ":legalize_tf_mlir",
        ":test_matchers",
        "//machina/compiler/jit",
        "//machina/compiler/mlir/machina",
        "//machina/compiler/mlir/machina:deserialize_mlir_module_utils",
        "//machina/compiler/tf2xla:xla_compiler",
        "//machina/compiler/tf2xla:xla_helpers",
        "//machina/core:framework",
        "//machina/core:test_main",
        "//machina/core/protobuf/tpu:compile_metadata_proto_cc",
        "//machina/core/tpu/kernels:tpu_compile_op_support",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Pass",
        "@local_xla//xla:shape_util",
        "@local_xla//xla/mlir_hlo:hlo_dialect_registration",
        "@local_xla//xla/tsl/platform:errors",
        "@stablehlo//:register",
    ],
)

cc_library(
    name = "legalize_tf_to_hlo",
    srcs = ["legalize_tf_to_hlo.cc"],
    hdrs = ["legalize_tf_to_hlo.h"],
    deps = [
        ":legalize_tf_mlir",
        "//machina/compiler/mlir/machina",
        "//machina/compiler/mlir/machina:deserialize_mlir_module_utils",
        "//machina/compiler/mlir/machina/transforms:set_tpu_infeed_layout",
        "//machina/compiler/mlir/tf2xla/api/v1:compile_tf_graph",
        "//machina/compiler/tf2xla:layout_util",
        "//machina/compiler/tf2xla:xla_helpers",
        "//machina/compiler/tf2xla:xla_tpu_backend_registration",
        "//machina/core:framework",
        "//machina/core/platform:status",
        "//machina/core/protobuf/tpu:compile_metadata_proto_cc",
        "//machina/core/tpu/kernels:tpu_compile_op_support",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:DataLayoutInterfaces",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Pass",
        "@local_xla//xla:shape_util",
        "@local_xla//xla/client:compile_only_client",
        "@local_xla//xla/mlir_hlo:hlo_dialect_registration",
        "@local_xla//xla/tsl/platform:errors",
        "@local_xla//xla/tsl/platform:statusor",
        "@stablehlo//:register",
    ],
)

tf_cc_test(
    name = "legalize_tf_to_hlo_test",
    srcs = ["legalize_tf_to_hlo_test.cc"],
    deps = [
        ":legalize_tf_to_hlo",
        ":test_matchers",
        "//machina/compiler/jit",
        "//machina/compiler/mlir/machina",
        "//machina/compiler/mlir/machina:deserialize_mlir_module_utils",
        "//machina/compiler/tf2xla:xla_compiler",
        "//machina/compiler/tf2xla:xla_helpers",
        "//machina/core:framework",
        "//machina/core:protos_all_cc",
        "//machina/core:test_main",
        "//machina/core/lib/monitoring:cell_reader",
        "//machina/core/protobuf:for_core_protos_cc",
        "//machina/core/protobuf/tpu:compile_metadata_proto_cc",
        "//machina/core/tpu/kernels:tpu_compile_op_support",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Pass",
        "@local_xla//xla:shape_util",
        "@local_xla//xla/client:client_library",
        "@local_xla//xla/mlir_hlo:hlo_dialect_registration",
        "@local_xla//xla/stream_executor:platform",
        "@local_xla//xla/stream_executor:platform_manager",
        "@local_xla//xla/tsl/platform:errors",
        "@stablehlo//:register",
    ],
)

cc_library(
    name = "clustering_bridge_passes",
    srcs = ["clustering_bridge_passes.cc"],
    hdrs = ["clustering_bridge_passes.h"],
    deps = [
        "//machina/compiler/jit:flags_headers",
        "//machina/compiler/mlir/machina/transforms:machina_passes",
        "//machina/compiler/mlir/machina/transforms:verify_no_outside_compilation_markers_pass",
        "//machina/compiler/mlir/machina/transforms/sparsecore:sparsecore_passes",
        "//machina/compiler/mlir/tf2xla/internal/passes:clustering_passes",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:Transforms",
    ],
)

tf_cc_test(
    name = "clustering_bridge_passes_test",
    srcs = ["clustering_bridge_passes_test.cc"],
    deps = [
        ":clustering_bridge_passes",
        "@com_google_googletest//:gtest_main",
        "@toolchain-project//mlir:Pass",
    ],
)

cc_library(
    name = "logging_hooks",
    srcs = ["logging_hooks.cc"],
    hdrs = ["logging_hooks.h"],
    deps = [
        "//machina/compiler/mlir/machina:bridge_logger",
        "//machina/core:framework",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:Pass",
    ],
)

tf_cc_test(
    name = "logging_hooks_test",
    srcs = ["logging_hooks_test.cc"],
    data = [
        "testdata/dead_const.mlir",
    ],
    deps = [
        ":logging_hooks",
        "//machina/compiler/mlir:register_common_dialects",
        "//machina/core:lib",
        "//machina/core:test",
        "//machina/core/platform:resource_loader",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Parser",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:Transforms",
        "@local_xla//xla/tsl/lib/core:status_test_util",
        "@local_xla//xla/tsl/platform:status",
    ],
)

cc_library(
    name = "mlir_bridge_pass_util",
    srcs = ["mlir_bridge_pass_util.cc"],
    hdrs = ["mlir_bridge_pass_util.h"],
    visibility = ["//machina/compiler/tf2xla:__pkg__"],
    deps = [
        "//machina/compiler/mlir/machina",
        "//machina/compiler/tf2xla:tf2xla_defs",
        "//machina/core:core_cpu_base",
        "//machina/core:framework",
        "//machina/core:protos_all_cc",
        "//machina/core/common_runtime:function_body",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:string_view",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Support",
        "@local_xla//xla/tsl/platform:status",
    ],
    alwayslink = 1,
)

tf_cc_test(
    name = "mlir_bridge_pass_util_test",
    srcs = ["mlir_bridge_pass_util_test.cc"],
    deps = [
        ":mlir_bridge_pass_util",
        "@com_google_googletest//:gtest_main",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Parser",
        "//machina/cc:array_ops",
        "//machina/cc:function_ops",
        "//machina/cc:functional_ops",
        "//machina/cc:ops",
        "//machina/cc:scope",
        "//machina/cc:tpu_ops",
        "//machina/compiler/mlir/machina",
        "//machina/compiler/tf2xla:tf2xla_defs",
        "//machina/core/protobuf:for_core_protos_cc",
        "//machina/core:core_cpu_base",
        "//machina/core:framework",
        "//machina/core:portable_gif_internal",
        "//machina/core:protos_all_cc",
        "//machina/core:test_main",
        "//machina/core:testlib",
        "//machina/core/platform:enable_tf2_utils",
        # "//machina/core/platform:resource_loader",
        "@local_xla//xla/tsl/lib/core:status_test_util",
    ],
)

tf_proto_library(
    name = "reproducer_proto",
    srcs = ["reproducer.proto"],
    protodeps = [
        "//machina/core:protos_all",
        "//machina/core/protobuf/tpu:compile_metadata_proto",
    ],
    visibility = [
        "//learning/brain/mlir/bridge:__pkg__",
        "//machina/compiler/mlir/tf2xla/api/v2:__pkg__",
    ],
)

cc_library(
    name = "node_order",
    srcs = ["node_order.cc"],
    hdrs = ["node_order.h"],
    deps = [
        "//machina/core:core_cpu_base",
        "//machina/core:lib",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
    ],
)

tf_cc_test(
    name = "node_order_test",
    size = "small",
    srcs = [
        "node_order_test.cc",
    ],
    deps = [
        ":node_order",
        "//machina/cc:cc_ops",
        "//machina/cc:cc_ops_internal",
        "//machina/cc:function_ops",
        "//machina/cc:sendrecv_ops",
        "//machina/core",
        "//machina/core:core_cpu",
        "//machina/core:core_cpu_internal",
        "//machina/core:direct_session_internal",
        "//machina/core:framework",
        "//machina/core:framework_internal",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core:ops",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core:testlib",
        "@com_google_absl//absl/log:check",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "graph_to_tf_executor_util",
    srcs = ["graph_to_tf_executor_util.cc"],
    hdrs = ["graph_to_tf_executor_util.h"],
    deps = [
        "//machina/core:core_cpu_base",
        "//machina/core:framework",
        "//machina/core:framework_types_hdr",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core/common_runtime:function_body",
        "//machina/core/platform:enable_tf2_utils",
        "//machina/core/protobuf:for_core_protos_cc",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:string_view",
    ],
)

tf_cc_test(
    name = "graph_to_tf_executor_util_test",
    srcs = ["graph_to_tf_executor_util_test.cc"],
    deps = [
        ":graph_to_tf_executor_util",
        "//machina/cc:array_ops",
        "//machina/cc:cc_ops",
        "//machina/cc:functional_ops",
        "//machina/cc:ops",
        "//machina/cc:scope",
        "//machina/cc:tpu_ops",
        "//machina/compiler/tf2xla/ops:xla_ops",
        "//machina/core:core_cpu_base",
        "//machina/core:framework",
        "//machina/core:portable_gif_internal",
        "//machina/core:protos_all_cc",
        "//machina/core/framework:tensor_testutil",
        "//machina/core/platform:enable_tf2_utils",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@local_xla//xla/tsl/lib/core:status_test_util",
        "@local_xla//xla/tsl/platform:status",
    ],
)

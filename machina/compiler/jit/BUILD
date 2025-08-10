load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda")
load("@local_config_rocm//rocm:build_defs.bzl", "if_rocm")
load("@local_xla//xla/stream_executor:build_defs.bzl", "if_cuda_or_rocm")
load(
    "@local_xla//xla/tsl:tsl.bzl",
    "check_deps",
    "internal_visibility",
)
load("//machina:machina.bzl", "if_libtpu", "if_with_tpu_support", "tf_cc_test", "tf_copts", "tf_cuda_cc_test", "tf_cuda_only_cc_test")
load("//machina:machina.default.bzl", "cc_header_only_library", "filegroup", "tf_custom_op_py_strict_library")
load("//machina/compiler/jit:package_groups.bzl", "legacy_jit_users_package_group")
load("//machina/core/platform:build_config.bzl", "tf_additional_all_protos", "tf_proto_library")
load(
    "//machina/core/platform:build_config_root.bzl",
    "if_static",
    "tf_cuda_tests_tags",
)
load("//machina/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = [
        ":internal",
        "//third_party/cloud_tpu/inference_converter:__pkg__",
        "//waymo/ml/deploy/system/aot_compilation:__pkg__",
    ],
    licenses = ["notice"],
)

legacy_jit_users_package_group(name = "legacy_jit_users")

package_group(
    name = "internal",
    includes = [
        "//machina/compiler/tf2xla:internal",
    ],
    packages = [
        "//machina/c/...",
        "//machina/compiler/tests/...",
        "//machina/core/tfrt/saved_model/...",
        "//machina/python/...",
    ],
)

package_group(
    name = "friends",
    includes = [
        "//machina/compiler/tf2xla:friends",
    ],
)

# defs.cc/h only contains string constants, and can be included in mobile
# builds.
filegroup(
    name = "mobile_srcs_no_runtime",
    srcs = [
        "defs.cc",
        "defs.h",
    ],
    visibility = [":friends"],
)

# Target that bundles up the XLA CPU and GPU JIT devices.
# This target is deprecated as it pulls in all the XLA devices and bloats the dependencies.
# Please use the individual targets in the deps list as needed. See b/336889334.
cc_library(
    name = "jit",
    visibility = internal_visibility([":legacy_jit_users"]),
    deps = [
        ":xla_cpu_device",
        ":xla_cpu_jit",
        "//machina/compiler/plugin",
    ] + if_cuda_or_rocm([
        ":xla_gpu_device",
        ":xla_gpu_jit",
    ]) + if_with_tpu_support([
        ":xla_tpu_device",
        ":xla_tpu_jit",
    ]),
    alwayslink = 1,
)

cc_library(
    name = "xla_cpu_jit",
    visibility = ["//visibility:public"],
    deps = [
        ":jit_compilation_passes",
        ":xla_kernel_creator",  # buildcleaner: keep
        "//machina/compiler/jit/kernels:xla_ops",
        "//machina/compiler/tf2xla/kernels:xla_dummy_ops",
        "//machina/compiler/tf2xla/kernels:xla_ops",
        "//machina/core/tfrt/common:pjrt_cpu_client_registration",
    ] + if_with_tpu_support(
        if_false = ["@local_xla//xla/service:cpu_plugin"],
        if_true = [],
    ),
    alwayslink = 1,
)

cc_library(
    name = "xla_gpu_jit",
    visibility = ["//visibility:public"],
    deps = if_cuda_or_rocm([
        ":jit_compilation_passes",
        ":xla_kernel_creator",  # buildcleaner: keep
        "//machina/compiler/jit/kernels:xla_ops",
        "//machina/compiler/tf2xla/kernels:xla_ops",
        "//machina/compiler/tf2xla/kernels:xla_dummy_ops",
        "@local_xla//xla/service:gpu_plugin",
        "//machina/core/tfrt/common:pjrt_gpu_client_registration",
    ]) + if_cuda([
        "@local_xla//xla/stream_executor/cuda:all_runtime",  # buildcleaner: keep
    ]) + if_rocm([
        "@local_xla//xla/stream_executor/rocm:all_runtime",  # buildcleaner: keep
    ]),
    alwayslink = 1,
)

cc_library(
    name = "xla_tpu_jit",
    visibility = ["//visibility:public"],
    deps = if_libtpu([
        "//machina/core/tpu/graph_rewrite:tpu_rewrite_pass_registration",
        "//machina/core/tpu/graph_rewrite:configure_tpu_embedding_rewrite_registration",
        "@local_xla//xla/stream_executor/tpu:tpu_transfer_manager",
    ]),
    alwayslink = 1,
)

cc_library(
    name = "xla_cpu_device",
    srcs = ["xla_cpu_device.cc"],
    visibility = [":friends"],
    deps = [
        ":common",
        ":flags",
        ":jit_compilation_passes",
        ":xla_device",
        ":xla_kernel_creator",  # buildcleaner: keep
        "//machina/compiler/jit/kernels:xla_ops",
        "//machina/compiler/tf2xla:layout_util",
        "//machina/compiler/tf2xla:xla_compiler",
        "//machina/compiler/tf2xla:xla_op_registry",
        "//machina/compiler/tf2xla/kernels:xla_ops",
        "//machina/core:core_cpu_internal",
        "//machina/core:lib",
        "@com_google_absl//absl/memory",
        "@local_xla//xla/stream_executor:platform_manager",
    ] + if_libtpu(
        if_false = [
            "@local_xla//xla/service:cpu_plugin",  # buildcleaner: keep
        ],
        if_true = [],
    ),
    alwayslink = 1,
)

cc_library(
    name = "xla_gpu_device",
    srcs = ["xla_gpu_device.cc"],
    visibility = [":friends"],
    deps = [
        ":common",
        ":flags",
        ":jit_compilation_passes",
        ":xla_device",
        ":xla_device_no_jit_rewrite_registration",
        ":xla_kernel_creator",  # buildcleaner: keep
        "//machina/compiler/jit/kernels:xla_ops",
        "//machina/compiler/tf2xla:layout_util",
        "//machina/compiler/tf2xla:xla_compiler",
        "//machina/compiler/tf2xla:xla_op_registry",
        "//machina/compiler/tf2xla/kernels:xla_ops",
        "//machina/core:core_cpu_internal",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@local_xla//xla/stream_executor:platform_manager",
        "@local_xla//xla/stream_executor/gpu:gpu_init",
    ] + if_libtpu(
        if_false = [
            "@local_xla//xla/service:gpu_plugin",  # buildcleaner: keep
        ],
        if_true = [],
    ),
    alwayslink = 1,
)

cc_library(
    name = "xla_tpu_device",
    srcs = ["xla_tpu_device.cc"],
    hdrs = ["xla_tpu_device.h"],
    visibility = [":friends"],
    deps = [
        ":xla_device",
        ":xla_device_context",
        ":xla_kernel_creator",  # buildcleaner: keep
        "//machina/compiler/jit/kernels:xla_ops",
        "//machina/compiler/tf2xla:common",
        "//machina/compiler/tf2xla:layout_util",
        "//machina/compiler/tf2xla:tf2xla_util",
        "//machina/compiler/tf2xla:xla_helpers",
        "//machina/compiler/tf2xla:xla_op_registry",
        "//machina/core:framework_internal",
        "//machina/core:lib_proto_parsing",
        "//machina/core:protos_all_cc",
        "//machina/core:session_options",
        "//machina/core/common_runtime:device",
        "//machina/core/common_runtime:device_factory",
        "//machina/core/common_runtime:dma_helper",
        "//machina/core/platform:status",
        "//machina/core/tpu:tpu_defs",
        "//machina/core/tpu:tpu_node_device_util",
        "//machina/core/tpu:virtual_device",
        "@com_google_absl//absl/types:optional",
        "@local_tsl//tsl/platform:statusor",
        "@local_xla//xla/stream_executor/tpu:c_api_conversions",
        "@local_xla//xla/stream_executor/tpu:status_helper",
        "@local_xla//xla/stream_executor/tpu:tpu_api",
        "@local_xla//xla/stream_executor/tpu:tpu_executor_base",
        "@local_xla//xla/stream_executor/tpu:tpu_node_context",
        "@local_xla//xla/stream_executor/tpu:tpu_platform_interface",
        "@local_xla//xla/stream_executor/tpu:tpu_stream_interface",
    ] + if_static([
        "//machina/core/common_runtime:copy_tensor",
        ":jit_compilation_passes",
    ]),
    alwayslink = 1,
)

cc_library(
    name = "xla_tensor",
    srcs = ["xla_tensor.cc"],
    hdrs = ["xla_tensor.h"],
    visibility = [":friends"],
    deps = [
        "//machina/compiler/tf2xla:common",
        "//machina/core:core_cpu_internal",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "@com_google_absl//absl/memory",
        "@local_xla//xla:shape_util",
        "@local_xla//xla/client:local_client",
        "@local_xla//xla/service:shaped_buffer",
    ],
)

cc_library(
    name = "xla_device_context",
    srcs = ["xla_device_context.cc"],
    hdrs = ["xla_device_context.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":xla_launch_util",
        ":xla_tensor",
        "//machina/compiler/tf2xla:common",
        "//machina/compiler/tf2xla:layout_util",
        "//machina/compiler/tf2xla:xla_compiler",
        "//machina/compiler/tf2xla:xla_helpers",
        "//machina/core:framework",
        "//machina/core:framework_internal",
        "//machina/core:lib_proto_parsing",
        "//machina/core/common_runtime:device",
        "//machina/core/common_runtime:dma_helper",
        "//machina/core/framework:allocator",
        "@com_google_absl//absl/synchronization",
        "@local_tsl//tsl/platform:statusor",
        "@local_xla//xla:util",
        "@local_xla//xla/client:local_client",
    ],
)

cc_library(
    name = "xla_device_no_jit_rewrite_registration",
    srcs = [
        "xla_compile_on_demand_op.cc",
        "xla_compiler_options_util.cc",
        "xla_device.cc",
        "xla_device_ops.cc",
        "xla_ops_on_regular_devices.cc",
        "xla_platform_info.cc",
    ],
    hdrs = [
        "xla_compile_on_demand_op.h",
        "xla_compiler_options_util.h",
        "xla_device.h",
        "xla_device_ops.h",
        "xla_platform_info.h",
    ],
    # Public visibility is needed for external TF/XLA backends.
    visibility = ["//visibility:public"],
    deps = [
        ":common",
        ":device_compilation_cluster_signature",
        ":device_compilation_profiler",
        ":device_compiler",
        ":device_executable_persistor",
        ":flags_headers",
        ":pjrt_base_device",
        ":pjrt_device_compiler_client",
        ":pjrt_device_context",
        ":variable_info",
        ":variable_info_util",
        ":xla_compile_util",
        ":xla_device_compiler_client",
        ":xla_device_context",
        ":xla_launch_util",
        ":xla_tensor",
        "//machina/compiler/jit/ops:xla_ops",
        "//machina/compiler/tf2xla:common",
        "//machina/compiler/tf2xla:layout_util",
        "//machina/compiler/tf2xla:tf2xla_util",
        "//machina/compiler/tf2xla:xla_compiler",
        "//machina/compiler/tf2xla:xla_op_registry",
        "//machina/compiler/tf2xla/kernels:xla_dummy_ops",
        "//machina/compiler/tf2xla/kernels:xla_ops",
        "//machina/core:array_ops_op_lib",
        "//machina/core:control_flow_ops_op_lib",
        "//machina/core:core_cpu",
        "//machina/core:core_cpu_internal",
        "//machina/core:dataset_ops_op_lib",
        "//machina/core:framework",
        "//machina/core:framework_internal",
        "//machina/core:functional_ops_op_lib",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core:math_ops_op_lib",
        "//machina/core:nn_ops_op_lib",
        "//machina/core:no_op_op_lib",
        "//machina/core:protos_all_cc",
        "//machina/core:resource_variable_ops_op_lib",
        "//machina/core:sendrecv_ops_op_lib",
        "//machina/core:state_ops_op_lib",
        "//machina/core/kernels:constant_op",
        "//machina/core/kernels:fifo_queue",
        "//machina/core/kernels:function_ops",
        "//machina/core/kernels:identity_op",
        "//machina/core/kernels:resource_variable_ops",
        "//machina/core/kernels:shape_ops",
        "//machina/core/kernels:variable_ops",
        "//machina/core/kernels/data:finalize_dataset_op",
        "//machina/core/kernels/data:generator_dataset_op",
        "//machina/core/kernels/data:iterator_ops",
        "//machina/core/kernels/data:optional_ops",
        "//machina/core/kernels/data:options_dataset_op",
        "//machina/core/kernels/data:prefetch_dataset_op",
        "//machina/core/platform:stream_executor_no_cuda",
        "//machina/core/profiler/lib:traceme",
        "//machina/core/tfrt/common:async_value_tensor",
        "//machina/core/tfrt/common:create_pjrt_client_util",
        "//machina/core/tfrt/common:global_state",
        "//machina/core/tfrt/common:pjrt_util",
        "//machina/core/tpu:tpu_defs",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
        "@local_xla//xla:executable_run_options",
        "@local_xla//xla/client:client_library",
        "@local_xla//xla/client:local_client",
        "@local_xla//xla/hlo/ir:hlo",
        "@local_xla//xla/pjrt:pjrt_client",
        "@local_xla//xla/pjrt:tf_pjrt_client",
        "@local_xla//xla/service:compiler",
        "@local_xla//xla/service:executable",
        "@local_xla//xla/service:stream_pool",
        "@local_xla//xla/service/gpu:gpu_executable_run_options",
        "@local_xla//xla/stream_executor:platform_manager",
        "@local_xla//xla/stream_executor:stream",
        "@local_xla//xla/stream_executor/host:host_platform_id",
        "@local_xla//xla/stream_executor/integrations:tf_allocator_adapter",
        "@local_xla//xla/tsl/framework:device_id_utils",
    ],
    alwayslink = 1,
)

check_deps(
    name = "xla_device_check_deps",
    disallowed_deps = [
        "@local_xla//xla/service/gpu:gpu_compiler",
        "@local_xla//xla/service/cpu:cpu_compiler",
    ],
    deps = [":xla_device"],
)

cc_library(
    name = "xla_device",
    hdrs = [
        "xla_compile_on_demand_op.h",
        "xla_device.h",
        "xla_device_ops.h",
    ],
    # Public visibility is needed for external TF/XLA backends.
    visibility = ["//visibility:public"],
    deps = [
        ":device_compilation_cluster_signature",
        ":device_compilation_profiler",
        ":jit_compilation_passes",
        ":variable_info",
        ":variable_info_util",
        ":xla_device_no_jit_rewrite_registration",
        ":xla_launch_util",
        ":xla_tensor",
        "//machina/compiler/jit/ops:xla_ops",
        "//machina/compiler/tf2xla:layout_util",
        "//machina/compiler/tf2xla:xla_compiler",
        "//machina/compiler/tf2xla:xla_op_registry",
        "//machina/compiler/tf2xla/kernels:xla_dummy_ops",
        "//machina/compiler/tf2xla/kernels:xla_ops",
        "//machina/core:array_ops_op_lib",
        "//machina/core:control_flow_ops_op_lib",
        "//machina/core:core_cpu",
        "//machina/core:core_cpu_internal",
        "//machina/core:dataset_ops_op_lib",
        "//machina/core:framework",
        "//machina/core:framework_internal",
        "//machina/core:functional_ops_op_lib",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core:math_ops_op_lib",
        "//machina/core:nn_ops_op_lib",
        "//machina/core:no_op_op_lib",
        "//machina/core:resource_variable_ops_op_lib",
        "//machina/core:sendrecv_ops_op_lib",
        "//machina/core:state_ops_op_lib",
        "//machina/core/kernels:constant_op",
        "//machina/core/kernels:fifo_queue",
        "//machina/core/kernels:function_ops",
        "//machina/core/kernels:identity_op",
        "//machina/core/kernels:resource_variable_ops",
        "//machina/core/kernels:shape_ops",
        "//machina/core/kernels:variable_ops",
        "//machina/core/kernels/data:finalize_dataset_op",
        "//machina/core/kernels/data:generator_dataset_op",
        "//machina/core/kernels/data:iterator_ops",
        "//machina/core/kernels/data:optional_ops",
        "//machina/core/kernels/data:options_dataset_op",
        "//machina/core/kernels/data:prefetch_dataset_op",
        "//machina/core/platform:stream_executor_no_cuda",
        "//machina/core/tfrt/common:async_value_tensor",
        "@com_google_absl//absl/types:optional",
        "@local_xla//xla/client:local_client",
        "@local_xla//xla/pjrt:pjrt_client",
    ],
)

cc_library(
    name = "shape_inference_helpers",
    srcs = ["shape_inference_helpers.cc"],
    hdrs = ["shape_inference_helpers.h"],
    visibility = [":friends"],
    deps = select({
        "//machina:android": [
            "//machina/core:portable_machina_lib",
        ],
        "//conditions:default": [
            "//machina/core:graph",
        ],
    }),
)

cc_library(
    name = "flags",
    srcs = ["flags.cc"],
    hdrs = ["flags.h"],
    visibility = [":friends"],
    deps = [
        "//machina/compiler/mlir/machina:dump_graph",
        "//machina/core:framework_internal",
        "//machina/core:framework_types_hdr",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core/tpu/kernels:sparse_core_xla_flags_defaults",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@local_xla//xla:parse_flags_from_env",
    ],
)

# Header-only version of "flags" library, for linking from the shared object
# without ODR violations.
cc_library(
    name = "flags_headers",
    hdrs = ["flags.h"],
    visibility = [":friends"],
    deps = [
        "//machina/core:framework_internal",
        "//machina/core:framework_types_hdr",
        "//machina/core:lib",
        "//machina/core/protobuf:for_core_protos_cc",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/types:optional",
    ],
)

cc_header_only_library(
    name = "flags_headers_only",
    features = [
        "-parse_headers",  # buildifier: disable=no-parse-headers
    ],
    deps = [":flags_headers"],
)

cc_library(
    name = "common",
    srcs = [
        "defs.cc",
    ],
    hdrs = [
        "defs.h",
    ],
    visibility = [":friends"],
)

# Internal targets below this point.

cc_library(
    name = "variable_info",
    srcs = ["variable_info.cc"],
    hdrs = ["variable_info.h"],
    visibility = [
        ":internal",
        # We reuse VariableInfo in TFRT's implementation of TpuExecuteOp.
        "//learning/brain/tfrt/tf_tpu:__pkg__",
        "//learning/brain/tfrt/tpu_common:__pkg__",
        "//machina/core/common_runtime/next_pluggable_device:__pkg__",
    ],
    deps = [
        "//machina/core:core_cpu_internal",
        "//machina/core:framework",
        "//machina/core:framework_internal",
        "//machina/core:lib",
        "//machina/core:lib_internal",
    ],
)

cc_library(
    name = "variable_info_util",
    srcs = ["variable_info_util.cc"],
    hdrs = ["variable_info_util.h"],
    visibility = [
        ":internal",
        # We reuse VariableInfo in TFRT's implementation of TpuExecuteOp.
        "//learning/brain/tfrt/tf_tpu:__pkg__",
    ],
    deps = [
        ":variable_info",
        "//machina/core:core_cpu_internal",
        "//machina/core:framework",
        "//machina/core:framework_internal",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "@com_google_absl//absl/algorithm:container",
    ],
)

cc_library(
    name = "pjrt_compile_util",
    srcs = ["pjrt_compile_util.cc"],
    hdrs = ["pjrt_compile_util.h"],
    visibility = [
        ":internal",
        "//machina/core/tfrt/gpu/kernel:__pkg__",
    ],
    deps = [
        ":device_compilation_profiler",
        ":device_compiler",
        ":xla_compile_util",
        ":xla_device_no_jit_rewrite_registration",
        "//machina/compiler/tf2xla:xla_compiler",
        "//machina/core:framework",
        "//machina/core:lib",
        "@local_tsl//tsl/platform:statusor",
        "@local_xla//xla/pjrt:pjrt_client",
    ],
)

tf_cuda_cc_test(
    name = "pjrt_compile_util_test",
    srcs = ["pjrt_compile_util_test.cc"],
    tags = [
        "config-cuda-only",
        "no_oss",  # This test only runs with GPU.
        "requires-gpu-nvidia",
        "xla",
    ],
    deps = [
        ":pjrt_compile_util",
        ":test_util",
        ":xla_gpu_jit",
        "//machina/cc:function_ops",
        "//machina/cc:math_ops",
        "//machina/cc:scope",
        "//machina/core:framework",
        "//machina/core:test",
        "//machina/core/kernels:ops_testutil",
        "//machina/core/platform:errors",
        "//machina/core/platform:status",
        "//machina/core/platform:status_matchers",
        "//machina/core/platform:statusor",
        "//machina/core/tfrt/common:pjrt_util",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "xla_launch_util",
    srcs = ["xla_launch_util.cc"],
    hdrs = ["xla_launch_util.h"],
    visibility = [
        ":internal",
        # We reuse VariableInfo in TFRT's implementation of TpuExecuteOp.
        "//learning/brain/tfrt/tf_tpu:__pkg__",
        "//learning/brain/tfrt/tpu_plugin:__pkg__",
        "//machina/core/tfrt/gpu/kernel:__pkg__",
    ],
    deps = [
        ":pjrt_tensor_buffer",
        ":pjrt_tensor_buffer_util",
        ":variable_info",
        ":variable_info_util",
        ":xla_tensor",
        "//machina/compiler/tf2xla:common",
        "//machina/compiler/tf2xla:xla_compiler",
        "//machina/core:core_cpu_internal",
        "//machina/core:framework",
        "//machina/core:framework_internal",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core:protos_all_cc",
        "//machina/core/common_runtime:dma_helper",
        "//machina/core/common_runtime:gpu_device_context",
        "//machina/core/common_runtime/gpu:gpu_serving_device_selector",
        "//machina/core/tfrt/common:async_value_tensor",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:span",
        "@local_xla//xla:shape_util",
        "@local_xla//xla:status_macros",
        "@local_xla//xla/client:local_client",
        "@local_xla//xla/pjrt:pjrt_client",
        "@local_xla//xla/pjrt:pjrt_common",
        "@local_xla//xla/pjrt:pjrt_future",
        "@local_xla//xla/service:shaped_buffer",
        "@local_xla//xla/stream_executor:device_memory_allocator",
        "@local_xla//xla/stream_executor:platform_manager",
        "@local_xla//xla/tsl/framework:device_id_utils",
        "@local_xla//xla/tsl/framework:serving_device_selector_policies",
    ],
)

tf_cc_test(
    name = "xla_launch_util_test",
    srcs = ["xla_launch_util_test.cc"],
    deps = [
        ":device_compiler",
        ":flags_headers",
        ":pjrt_device_compiler_client",
        ":variable_info",
        ":variable_info_util",
        ":xla_cpu_device",
        ":xla_cpu_jit",
        ":xla_device_no_jit_rewrite_registration",
        ":xla_launch_util",
        "//machina/compiler/tf2xla:xla_op_registry",
        "//machina/core:framework",
        "//machina/core/framework:fake_input",
        "//machina/core/framework:tensor_testutil",
        "//machina/core/kernels:ops_testutil",
        "//machina/core/platform:refcount",
        "//machina/core/tfrt/common:create_pjrt_client_util",
        "//machina/core/tfrt/common:pjrt_util",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:statusor",
        "@local_xla//xla/pjrt:pjrt_client",
        "@local_xla//xla/pjrt:pjrt_common",
        "@local_xla//xla/pjrt/plugin/xla_cpu:cpu_client_options",
        "@local_xla//xla/pjrt/plugin/xla_cpu:xla_cpu_pjrt_client",
        "@local_xla//xla/tests:literal_test_util",
        "@local_xla//xla/tsl/framework:device_id_utils",
        "@local_xla//xla/tsl/lib/core:status_test_util",
    ],
)

tf_cuda_only_cc_test(
    name = "xla_launch_util_gpu_test",
    srcs = ["xla_launch_util_gpu_test.cc"],
    deps = [
        ":device_compiler",
        ":flags_headers",
        ":pjrt_device_compiler_client",
        ":pjrt_device_context",
        ":variable_info",
        ":variable_info_util",
        ":xla_device_no_jit_rewrite_registration",
        ":xla_gpu_jit",
        ":xla_launch_util",
        "//machina/compiler/tf2xla:xla_op_registry",
        "//machina/core:framework",
        "//machina/core/framework:fake_input",
        "//machina/core/framework:tensor_testutil",
        "//machina/core/kernels:ops_testutil",
        "//machina/core/platform:refcount",
        "//machina/core/tfrt/common:create_pjrt_client_util",
        "//machina/core/tfrt/common:pjrt_util",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:statusor",
        "@local_xla//xla/pjrt:pjrt_client",
        "@local_xla//xla/tests:literal_test_util",
        "@local_xla//xla/tsl/lib/core:status_test_util",
    ],
)

cc_library(
    name = "xla_compile_util",
    srcs = ["xla_compile_util.cc"],
    hdrs = ["xla_compile_util.h"],
    visibility = [
        ":internal",
        "//machina/core/tfrt/gpu/kernel:__pkg__",
    ],
    deps = [
        ":flags_headers",
        "//machina/compiler/tf2xla:xla_argument",
        "//machina/core:core_cpu_base",
        "//machina/core:framework",
        "//machina/core/tfrt/common:global_state",
        "//machina/core/util:determinism",
        "@com_google_absl//absl/status",
    ],
)

tf_cc_test(
    name = "xla_compile_util_test",
    srcs = [
        "xla_compile_util_test.cc",
    ],
    deps = [
        ":flags_headers",
        ":xla_compile_util",
        "//machina/compiler/tf2xla:xla_compiler",
        "//machina/core:test_main",
        "//machina/core/framework:fake_input",
        "//machina/core/kernels:identity_op",
        "//machina/core/kernels:ops_testutil",
        "//machina/core/tpu:tpu_defs",
        "@com_google_googletest//:gtest",
    ],
)

tf_proto_library(
    name = "xla_compilation_cache_proto",
    srcs = ["xla_compilation_cache.proto"],
    protodeps = tf_additional_all_protos() + ["@local_xla//xla/service:hlo_proto"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "device_compiler",
    hdrs = ["device_compiler.h"],
    copts = tf_copts(),
    visibility = [":internal"],
    deps = [
        ":device_compilation_cache",
        ":device_compilation_cluster_signature",
        ":device_compilation_profiler",
        ":device_compiler_client",
        ":device_executable_persistor",
        ":flags_headers",
        ":tf_graph_to_hlo_compiler",
        ":xla_compile_util",
        "//machina/compiler/tf2xla:xla_compiler",
        "//machina/core:framework",
        "//machina/core:framework_lite",
        "//machina/core:lib",
        "//machina/core/framework:resource_base",
        "//machina/core/platform:thread_annotations",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:span",
    ],
)

tf_cc_test(
    name = "device_compiler_disable_test",
    srcs = [
        "device_compiler_disable_test.cc",
    ],
    tags = ["no_cuda_on_cpu_tap"],
    deps = [
        ":device_compilation_profiler",
        ":device_compiler",
        ":flags",
        ":xla_cpu_jit",
        ":xla_device_compiler_client",
        "//machina/compiler/tf2xla:xla_compiler",
        "//machina/core:test",
        "//machina/core:test_main",
        "@local_xla//xla/client:client_library",
    ],
)

cc_library(
    name = "jit_compilation_passes",
    srcs = ["jit_compilation_pass_registration.cc"],
    deps = [
        ":compilation_passes",
        ":xla_activity_logging_listener",
        "//machina/compiler/tf2xla:functionalize_control_flow_pass_registration",
        "//machina/compiler/tf2xla:mlir_bridge_pass_registration",
        "//machina/core:core_cpu_internal",
    ],
    alwayslink = 1,
)

cc_library(
    name = "get_compiler_ir",
    srcs = ["get_compiler_ir.cc"],
    hdrs = ["get_compiler_ir.h"],
    visibility = [":internal"],
    deps = [
        ":compilability_check_util",
        ":device_compiler",
        ":variable_info",
        ":variable_info_util",
        ":xla_device_no_jit_rewrite_registration",
        ":xla_launch_util",
        "//machina/compiler/tf2xla:xla_compiler",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core/common_runtime:core_cpu_internal",
        "//machina/core/common_runtime/eager:tensor_handle",
        "//machina/core/platform:refcount",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@local_xla//xla:status_macros",
        "@local_xla//xla/client:executable_build_options",
        "@local_xla//xla/client:local_client",
        "@local_xla//xla/hlo/translate:portable_api",
        "@local_xla//xla/service:hlo_graph_dumper",
        "@local_xla//xla/stream_executor:platform",
        "@local_xla//xla/stream_executor/host:host_platform_id",
    ],
    alwayslink = 1,
)

# Header-only version of "flags" library, for linking from the shared object
# without ODR violations.
cc_library(
    name = "get_compiler_ir_hdrs",
    textual_hdrs = ["get_compiler_ir.h"],
    visibility = [":internal"],
    deps = [
        "//machina/compiler/tf2xla:xla_compiler",
        "//machina/core:framework",
        "//machina/core:protos_all_cc",
        "//machina/core/platform:statusor",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_header_only_library(
    name = "get_compiler_ir_hdrs_only",
    features = [
        "-parse_headers",  # buildifier: disable=no-parse-headers
    ],
    deps = [":get_compiler_ir_hdrs"],
)

# This target can be used by XLA device plugins to prevent circular dependencies, and provides access to all of the required headers for building a device library.
cc_header_only_library(
    name = "xla_jit_headers_lib",
    visibility = ["//visibility:public"],
    deps = [
        ":xla_cpu_device",
        ":xla_cpu_jit",
        ":xla_gpu_device",
        ":xla_gpu_jit",
    ],
)

cc_library(
    name = "xla_kernel_creator",
    srcs = [
        "xla_kernel_creator.cc",
    ],
    hdrs = [
        "xla_kernel_creator.h",
    ],
    visibility = [
        ":internal",
        "//machina/core/common_runtime/eager:__pkg__",
    ],
    deps = [
        ":common",
        ":compilability_check_util",
        ":compilation_passes",
        ":flags",
        ":jit_compilation_passes",
        "//machina/compiler/jit/kernels:xla_ops_no_jit_rewrite_registration",
        "//machina/compiler/tf2xla:mlir_bridge_pass",
        "//machina/compiler/tf2xla:xla_compiler",
        "//machina/compiler/tf2xla:xla_op_registry",
        "//machina/core:core_cpu_internal",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
    alwayslink = 1,
)

tf_cc_test(
    name = "xla_kernel_creator_test",
    srcs = [
        "xla_kernel_creator.h",
        "xla_kernel_creator_test.cc",
    ],
    deps = [
        ":xla_kernel_creator",
        "//machina/core:core_cpu_internal",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core:session_options",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core:testlib",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
    ],
)

cc_library(
    name = "resource_operation_safety_analysis",
    srcs = ["resource_operation_safety_analysis.cc"],
    hdrs = ["resource_operation_safety_analysis.h"],
    deps = [
        ":xla_cluster_util",
        "//machina/compiler/tf2xla:resource_operation_table",
        "//machina/core:framework",
        "//machina/core:graph",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@local_xla//xla/service/graphcycles",
    ],
)

tf_cc_test(
    name = "resource_operation_safety_analysis_test",
    srcs = ["resource_operation_safety_analysis_test.cc"],
    deps = [
        ":common",
        ":resource_operation_safety_analysis",
        "//machina/cc:cc_ops",
        "//machina/cc:cc_ops_internal",
        "//machina/cc:function_ops",
        "//machina/cc:functional_ops",
        "//machina/cc:ops",
        "//machina/cc:resource_variable_ops",
        "//machina/cc:sendrecv_ops",
        "//machina/compiler/jit/kernels:xla_ops",
        "//machina/compiler/tf2xla:xla_compiler",
        "//machina/compiler/tf2xla/kernels:xla_ops",
        "//machina/core:core_cpu",
        "//machina/core:framework",
        "//machina/core:framework_internal",
        "//machina/core:graph",
        "//machina/core:lib",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core:testlib",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "shape_inference",
    srcs = ["shape_inference.cc"],
    hdrs = ["shape_inference.h"],
    visibility = [":friends"],
    deps = [
        ":shape_inference_helpers",
        "//machina/core:core_cpu_internal",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "test_util",
    testonly = 1,
    srcs = ["test_util.cc"],
    hdrs = ["test_util.h"],
    deps = [
        ":shape_inference",
        "//machina/core:core_cpu",
        "//machina/core:core_cpu_base",
        "//machina/core:framework",
        "//machina/core:lib",
        "@local_xla//xla:status_macros",
    ],
)

tf_cc_test(
    name = "shape_inference_test",
    srcs = ["shape_inference_test.cc"],
    deps = [
        ":shape_inference",
        ":test_util",
        "//machina/cc:cc_ops",
        "//machina/cc:cc_ops_internal",
        "//machina/cc:ops",
        "//machina/cc:resource_variable_ops",
        "//machina/cc:scope",
        "//machina/core:core_cpu_base",
        "//machina/core:framework",
        "//machina/core:ops",
        "//machina/core:portable_gif_internal",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core/kernels:constant_op",
        "@local_tsl//tsl/platform:status",
    ],
)

cc_library(
    name = "encapsulate_util",
    srcs = ["encapsulate_util.cc"],
    hdrs = ["encapsulate_util.h"],
    deps = [
        ":shape_inference",
        "//machina/compiler/tf2xla:tf2xla_util",
        "//machina/core:framework",
        "//machina/core:graph",
        "//machina/core:protos_all_cc",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
    ],
)

tf_cc_test(
    name = "encapsulate_util_test",
    srcs = ["encapsulate_util_test.cc"],
    deps = [
        ":encapsulate_util",
        "//machina/cc:cc_ops",
        "//machina/cc:scope",
        "//machina/core:framework",
        "//machina/core:ops",
        "//machina/core:protos_all_cc",
        "//machina/core:test",
        "//machina/core:test_main",
    ],
)

cc_library(
    name = "compilation_passes",
    srcs = [
        "build_xla_ops_pass.cc",
        "clone_constants_for_better_clustering.cc",
        "cluster_scoping_pass.cc",
        "deadness_analysis.cc",
        "deadness_analysis_internal.h",
        "encapsulate_subgraphs_pass.cc",
        "encapsulate_xla_computations_pass.cc",
        "extract_outside_compilation_pass.cc",
        "force_xla_constants_on_host_pass.cc",
        "increase_dynamism_for_auto_jit_pass.cc",
        "mark_for_compilation_pass.cc",
        "mark_for_compilation_pass_test_helper.cc",
        "partially_decluster_pass.cc",
        "report_clustering_info_pass.cc",
    ],
    hdrs = [
        "build_xla_ops_pass.h",
        "clone_constants_for_better_clustering.h",
        "cluster_scoping_pass.h",
        "deadness_analysis.h",
        "encapsulate_subgraphs_pass.h",
        "encapsulate_xla_computations_pass.h",
        "extract_outside_compilation_pass.h",
        "force_xla_constants_on_host_pass.h",
        "increase_dynamism_for_auto_jit_pass.h",
        "mark_for_compilation_pass.h",
        "mark_for_compilation_pass_test_helper.h",
        "partially_decluster_pass.h",
        "report_clustering_info_pass.h",
    ],
    visibility = [
        ":internal",
        "//third_party/cloud_tpu/inference_converter:__pkg__",
        "//waymo/onboard/ml/chauffeur_net:__pkg__",
    ],
    deps = [
        "compilability_check_util",
        ":common",
        ":device_util",
        ":encapsulate_util",
        ":flags",
        ":resource_operation_safety_analysis",
        ":shape_inference_helpers",
        ":xla_activity_listener",
        ":xla_cluster_util",
        "//machina/cc:cc_ops",
        "//machina/cc:functional_ops",
        "//machina/cc:ops",
        "//machina/cc:scope",
        "//machina/cc:scope_internal",
        "//machina/compiler/jit/ops:xla_ops",
        "//machina/compiler/tf2xla:resource_operation_table",
        "//machina/compiler/tf2xla:side_effect_util",
        "//machina/compiler/tf2xla:tf2xla_util",
        "//machina/compiler/tf2xla:xla_compiler",
        "//machina/compiler/tf2xla:xla_op_registry",
        "//machina/compiler/tf2xla/cc:xla_jit_ops",
        "//machina/compiler/tf2xla/cc:xla_ops",
        "//machina/core:core_cpu",
        "//machina/core:core_cpu_internal",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core:protos_all_cc",
        "//machina/core/framework:bounds_check",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@local_xla//xla:status_macros",
        "@local_xla//xla:union_find",
        "@local_xla//xla:util",
        "@local_xla//xla:xla_data_proto_cc",
        "@local_xla//xla/service/graphcycles",
    ],
)

cc_library(
    name = "xla_cluster_util",
    srcs = ["xla_cluster_util.cc"],
    hdrs = ["xla_cluster_util.h"],
    deps = [
        ":flags",
        ":xla_activity_proto_cc",
        "//machina/core:core_cpu",
        "//machina/core:framework",
        "//machina/core:framework_internal",
        "//machina/core:graph",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core/framework:bounds_check",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@local_xla//xla:status_macros",
        "@local_xla//xla/service/graphcycles",
    ],
)

cc_library(
    name = "device_util",
    srcs = ["device_util.cc"],
    hdrs = ["device_util.h"],
    deps = [
        "//machina/compiler/tf2xla:xla_compiler",
        "//machina/compiler/tf2xla:xla_op_registry",
        "//machina/core:framework",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/numeric:bits",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@local_xla//xla:status_macros",
    ],
)

tf_cc_test(
    name = "device_util_test",
    srcs = ["device_util_test.cc"],
    deps = [
        ":device_util",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core:testlib",
    ],
)

tf_cc_test(
    name = "deadness_analysis_test",
    size = "small",
    srcs = [
        "deadness_analysis_internal.h",
        "deadness_analysis_test.cc",
    ],
    deps = [
        ":common",
        ":compilation_passes",
        "//machina/cc:cc_ops",
        "//machina/cc:cc_ops_internal",
        "//machina/cc:function_ops",
        "//machina/cc:ops",
        "//machina/cc:sendrecv_ops",
        "//machina/compiler/jit/kernels:xla_ops",
        "//machina/compiler/tf2xla:xla_compiler",
        "//machina/compiler/tf2xla/kernels:xla_ops",
        "//machina/core:core_cpu",
        "//machina/core:framework",
        "//machina/core:framework_internal",
        "//machina/core:graph",
        "//machina/core:lib",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core:testlib",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)

cc_library(
    name = "compilation_passes_test_main",
    testonly = True,
    srcs = ["compilation_passes_test_main.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":flags",
        "//machina/core:lib",
        "//machina/core:test",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "compilation_passes_test",
    size = "small",
    srcs = [
        "build_xla_ops_pass_test.cc",
        "clone_constants_for_better_clustering_test.cc",
        "cluster_scoping_pass_test.cc",
        "encapsulate_subgraphs_pass_test.cc",
        "encapsulate_xla_computations_pass_test.cc",
        "extract_outside_compilation_pass_test.cc",
        "force_xla_constants_on_host_pass_test.cc",
        "increase_dynamism_for_auto_jit_pass_test.cc",
        "mark_for_compilation_pass_test.cc",
        "partially_decluster_pass_test.cc",
        "rearrange_function_argument_pass_test.cc",
    ],
    tags = [
        # TODO(b/141643254) Re-enable msan after fixing
        # use-of-uninitialized-value error.
        "nomsan",
    ] + tf_cuda_tests_tags(),
    deps = [
        ":common",
        ":compilability_check_util",
        ":compilation_passes",
        ":compilation_passes_test_main",
        ":encapsulate_util",
        ":flags",
        ":node_matchers",
        ":test_util",
        ":xla_cluster_util",
        ":xla_cpu_device",
        ":xla_gpu_device",
        "//machina/cc:cc_ops",
        "//machina/cc:cc_ops_internal",
        "//machina/cc:function_ops",
        "//machina/cc:functional_ops",
        "//machina/cc:ops",
        "//machina/cc:resource_variable_ops",
        "//machina/cc:scope",
        "//machina/cc:sendrecv_ops",
        "//machina/compiler/jit/kernels:xla_ops",
        "//machina/compiler/tf2xla:rearrange_function_argument",
        "//machina/compiler/tf2xla:side_effect_util",
        "//machina/compiler/tf2xla:test_util",
        "//machina/compiler/tf2xla:xla_compiler",
        "//machina/compiler/tf2xla/cc:xla_jit_ops",
        "//machina/compiler/tf2xla/cc:xla_ops",
        "//machina/compiler/tf2xla/kernels:xla_dummy_ops",
        "//machina/compiler/tf2xla/kernels:xla_ops",
        "//machina/core:all_kernels",
        "//machina/core:core_cpu",
        "//machina/core:framework",
        "//machina/core:framework_internal",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core:session_options",
        "//machina/core:test",
        "//machina/core:testlib",
        "//machina/core/common_runtime:device_set",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
        "@local_xla//xla/hlo/testlib:test",
    ],
)

tf_cc_test(
    name = "xla_cluster_util_test",
    size = "small",
    srcs = [
        "xla_cluster_util_test.cc",
    ],
    deps = [
        ":common",
        ":xla_cluster_util",
        "//machina/cc:cc_ops",
        "//machina/cc:cc_ops_internal",
        "//machina/cc:function_ops",
        "//machina/cc:functional_ops",
        "//machina/cc:ops",
        "//machina/compiler/jit/kernels:xla_ops",
        "//machina/compiler/tf2xla:xla_compiler",
        "//machina/compiler/tf2xla/kernels:xla_ops",
        "//machina/core:core_cpu",
        "//machina/core:core_cpu_lib",
        "//machina/core:framework",
        "//machina/core:framework_internal",
        "//machina/core:lib",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core:testlib",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/strings",
        "@local_xla//xla:status_macros",
    ],
)

cc_library(
    name = "node_matchers",
    testonly = True,
    srcs = ["node_matchers.cc"],
    hdrs = ["node_matchers.h"],
    deps = [
        "//machina/cc:ops",
        "//machina/core:framework",
        "//machina/core:graph",
        "//machina/core:protos_all_cc",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
        "@local_xla//xla/hlo/testlib:test",
    ],
)

tf_cc_test(
    name = "node_matchers_test",
    srcs = ["node_matchers_test.cc"],
    deps = [
        ":node_matchers",
        "//machina/cc:cc_ops",
        "//machina/cc:cc_ops_internal",
        "//machina/cc:ops",
        "//machina/core:ops",
        "//machina/core:test_main",
    ],
)

cc_library(
    name = "compilability_check_util",
    srcs = ["compilability_check_util.cc"],
    hdrs = ["compilability_check_util.h"],
    visibility = [
        ":friends",
    ],
    deps = [
        ":common",
        ":device_util",
        ":flags",
        ":resource_operation_safety_analysis",
        ":xla_activity_listener",
        ":xla_activity_proto_cc",
        ":xla_cluster_util",
        "//machina/compiler/tf2xla:resource_operation_table",
        "//machina/compiler/tf2xla:tf2xla_util",
        "//machina/compiler/tf2xla:xla_compiler",
        "//machina/compiler/tf2xla:xla_op_registry",
        "//machina/core:core_cpu",
        "//machina/core:framework",
        "//machina/core:graph",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@local_xla//xla:union_find",
        "@local_xla//xla:util",
        "@local_xla//xla/service/graphcycles",
    ],
)

tf_cc_test(
    name = "compilability_check_util_test",
    srcs = ["compilability_check_util_test.cc"],
    deps = [
        ":compilability_check_util",
        ":xla_cpu_device",
        ":xla_cpu_jit",
        "//machina/cc:cc_ops",
        "//machina/cc:function_ops",
        "//machina/cc:functional_ops",
        "//machina/cc:ops",
        "//machina/cc:scope",
        "//machina/compiler/tf2xla:test_util",
        "//machina/compiler/tf2xla:xla_compiler",
        "//machina/compiler/tf2xla/cc:xla_jit_ops",
        "//machina/compiler/tf2xla/cc:xla_ops",
        "//machina/core:core_cpu",
        "//machina/core:framework",
        "//machina/core:ops",
        "//machina/core:protos_all_cc",
        "//machina/core:test",
        "//machina/core:test_main",
        "@com_google_absl//absl/memory",
    ],
)

tf_cc_test(
    name = "xla_activity_listener_test",
    srcs = ["xla_activity_listener_test.cc"],
    deps = [
        ":flags",
        ":xla_activity_listener",
        ":xla_cpu_device",
        ":xla_cpu_jit",
        "//machina/cc:cc_ops",
        "//machina/cc:ops",
        "//machina/core:all_kernels",
        "//machina/core:core_cpu",
        "//machina/core:framework",
        "//machina/core:ops",
        "//machina/core:test",
        "//machina/core/common_runtime:direct_session_internal",
        "//machina/core/kernels:cwise_op",
        "//machina/core/kernels:matmul_op",
        "//machina/core/kernels:partitioned_function_ops",
    ],
)

tf_custom_op_py_strict_library(
    name = "xla_ops_py",
    kernels = ["//machina/compiler/jit/ops:xla_ops"],
    visibility = [
        ":friends",
    ],
    deps = [
        "//machina/compiler/jit/ops:xla_ops_grad",
        "//machina/compiler/jit/ops:xla_ops_wrapper_py",
    ],
)

cc_library(
    name = "xla_activity_listener",
    srcs = ["xla_activity_listener.cc"],
    hdrs = ["xla_activity_listener.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":xla_activity_proto_cc",
        "//machina/core:lib",
        "@com_google_absl//absl/synchronization",
    ],
)

tf_proto_library(
    name = "xla_activity_proto",
    srcs = ["xla_activity.proto"],
    protodeps = tf_additional_all_protos(),
)

cc_library(
    name = "xla_activity_logging_listener",
    srcs = ["xla_activity_logging_listener.cc"],
    deps = [
        ":xla_activity_listener",
        ":xla_activity_proto_cc",
        "//machina/core:lib",
        "@com_google_absl//absl/memory",
    ],
    alwayslink = 1,
)

cc_library(
    name = "tf_to_hlo_compiler",
    hdrs = ["tf_to_hlo_compiler.h"],
    deps = [
        "//machina/compiler/tf2xla:xla_compiler",
        "//machina/core:framework",
    ],
)

cc_library(
    name = "tf_graph_to_hlo_compiler",
    srcs = ["tf_graph_to_hlo_compiler.cc"],
    hdrs = ["tf_graph_to_hlo_compiler.h"],
    deps = [
        ":tf_to_hlo_compiler",
        "//machina/compiler/tf2xla:xla_argument",
        "//machina/compiler/tf2xla:xla_compiler",
        "//machina/core:framework",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "device_compilation_profiler",
    srcs = ["device_compilation_profiler.cc"],
    hdrs = ["device_compilation_profiler.h"],
    deps = [
        ":xla_activity_listener",
        ":xla_activity_proto_cc",
        ":xla_compile_util",
        "//machina/core:framework",
        "//machina/core:protos_all_cc",
        "//machina/core/platform:errors",
        "//machina/core/platform:status",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:mutex",
    ],
)

cc_library(
    name = "device_compiler_client",
    srcs = ["device_compiler_client.cc"],
    hdrs = ["device_compiler_client.h"],
    visibility = [":internal"],
    deps = [
        "//machina/compiler/tf2xla:xla_compiler",
        "//machina/core/util:determinism",
        "@local_xla//xla/client:executable_build_options",
    ],
)

tf_cc_test(
    name = "device_compiler_client_test",
    srcs = ["device_compiler_client_test.cc"],
    deps = [
        ":device_compiler_client",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "xla_device_compiler_client",
    srcs = ["xla_device_compiler_client.cc"],
    hdrs = ["xla_device_compiler_client.h"],
    deps = [
        ":device_compiler_client",
        "@local_xla//xla/client:local_client",
    ],
)

cc_library(
    name = "device_executable_persistor",
    srcs = ["device_executable_persistor.cc"],
    hdrs = ["device_executable_persistor.h"],
    deps = [
        ":device_compiler_client",
        ":xla_compilation_cache_proto_cc",
        ":xla_device_compiler_client",
        "//machina/compiler/tf2xla:xla_compiler",
        "//machina/core:core_cpu_base",
        "//machina/core:framework",
        "//machina/core:lib_headers_for_pybind",
        "//machina/core:portable_gif_internal",
        "//machina/core/platform:path",
        "//machina/core/platform:status",
        "//machina/core/platform:statusor",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:statusor",
        "@local_xla//xla:util",
        "@local_xla//xla/pjrt:pjrt_client",
        "@local_xla//xla/service:hlo_proto_cc",
    ],
)

cc_library(
    name = "device_compilation_cache",
    hdrs = ["device_compilation_cache.h"],
    deps = [
        ":device_compilation_cluster_signature",
        ":xla_compile_util",
        "//machina/compiler/tf2xla:xla_compiler",
        "//machina/core:framework_lite",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@local_xla//xla/client:local_client",
        "@local_xla//xla/pjrt:pjrt_client",
    ],
)

cc_library(
    name = "device_compilation_cluster_signature",
    srcs = ["device_compilation_cluster_signature.cc"],
    hdrs = ["device_compilation_cluster_signature.h"],
    deps = [
        "//machina/compiler/tf2xla:xla_compiler",
        "//machina/core:framework",
        "//machina/core/platform:types",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "pjrt_device_compiler_client",
    srcs = ["pjrt_device_compiler_client.cc"],
    hdrs = ["pjrt_device_compiler_client.h"],
    deps = [
        ":device_compiler_client",
        "@local_xla//xla/pjrt:pjrt_client",
    ],
)

cc_library(
    name = "pjrt_base_device",
    srcs = ["pjrt_base_device.cc"],
    hdrs = ["pjrt_base_device.h"],
    # Public visibility is needed for external TF/XLA backends.
    visibility = ["//visibility:public"],
    deps = [
        "//machina/compiler/tf2xla:layout_util",
        "//machina/core:framework",
        "//machina/core/common_runtime:local_device",
    ],
)

cc_library(
    name = "pjrt_tensor_buffer",
    hdrs = ["pjrt_tensor_buffer.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//machina/core:framework",
        "//machina/core:protos_all_cc",
        "@local_xla//xla/pjrt:pjrt_client",
    ],
)

cc_library(
    name = "pjrt_tensor_buffer_util",
    srcs = ["pjrt_tensor_buffer_util.cc"],
    hdrs = ["pjrt_tensor_buffer_util.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":pjrt_tensor_buffer",
        "//machina/core:framework",
        "//machina/core/common_runtime:dma_helper",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@local_tsl//tsl/platform:statusor",
        "@local_xla//xla/pjrt:pjrt_client",
    ],
)

tf_cuda_only_cc_test(
    name = "pjrt_tensor_buffer_util_test",
    srcs = ["pjrt_tensor_buffer_util_test.cc"],
    tags = tf_cuda_tests_tags() + [
        "config-cuda-only",
        "no_oss",
    ],
    deps = [
        ":pjrt_tensor_buffer_util",
        ":test_util",
        ":xla_gpu_jit",
        "//machina/core:framework",
        "//machina/core:gpu_runtime",  # Needed to register DEVICE_GPU
        "//machina/core/tfrt/common:pjrt_util",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:statusor",
        "@local_xla//xla:shape_util",
        "@local_xla//xla/pjrt:pjrt_client",
    ],
)

cc_library(
    name = "pjrt_device_context",
    srcs = [
        "pjrt_device_context.cc",
    ],
    hdrs = [
        "pjrt_device_context.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":pjrt_tensor_buffer",
        ":pjrt_tensor_buffer_util",
        "//machina/c/experimental/next_pluggable_device:tensor_pjrt_buffer_util",
        "//machina/compiler/tf2xla:common",
        "//machina/compiler/tf2xla:layout_util",
        "//machina/core:framework",
        "//machina/core/common_runtime:dma_helper",
        "//machina/core/common_runtime/next_pluggable_device:next_pluggable_device_api",
        "//machina/core/platform:status",
        "//machina/core/profiler/lib:traceme",
        "//machina/core/tfrt/common:async_value_tensor",
        "//machina/core/tfrt/common:create_pjrt_client_util",
        "@com_google_absl//absl/status",
        "@local_xla//xla/pjrt:pjrt_client",
        "@local_xla//xla/pjrt:pjrt_common",
        "@local_xla//xla/tsl/c:tsl_status_internal",
        "@local_xla//xla/tsl/framework:device_id_utils",
    ],
)

cc_library(
    name = "xla_host_recv_device_context",
    srcs = [
        "xla_host_recv_device_context.cc",
    ],
    hdrs = [
        "xla_host_recv_device_context.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//machina/compiler/tf2xla:common",
        "//machina/core:framework",
        "@local_xla//xla:shape_util",
        "@local_xla//xla/stream_executor:device_memory",
        "@local_xla//xla/stream_executor:stream",
        "@tf_runtime//:async_value",
    ],
)

cc_library(
    name = "xla_host_send_device_context",
    srcs = [
        "xla_host_send_device_context.cc",
    ],
    hdrs = [
        "xla_host_send_device_context.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//machina/compiler/tf2xla:common",
        "//machina/core:framework",
        "@local_xla//xla:shape_util",
        "@local_xla//xla/stream_executor:device_memory",
        "@local_xla//xla/stream_executor:stream",
        "@tf_runtime//:async_value",
    ],
)

tf_cuda_only_cc_test(
    name = "xla_host_send_recv_device_context_test",
    srcs = ["xla_host_send_recv_device_context_test.cc"],
    tags = tf_cuda_tests_tags() + [
        "config-cuda-only",
        "no_oss",  # Temporarily disable OSS.
    ],
    deps = [
        ":flags",
        ":xla_device",
        ":xla_gpu_device",
        ":xla_host_recv_device_context",
        ":xla_host_send_device_context",
        "//machina/compiler/tf2xla:common",
        "//machina/compiler/tf2xla:xla_op_registry",
        "//machina/core:framework_internal",
        "//machina/core:gpu_runtime",  # Needed to register DEVICE_GPU
        "//machina/core:test",
        "//machina/core/framework:tensor_testutil",
        "@com_google_googletest//:gtest_main",
        "@local_xla//xla:shape_util",
        "@local_xla//xla/stream_executor:device_memory",
        "@local_xla//xla/stream_executor:platform_manager",
        "@local_xla//xla/stream_executor:stream",
        "@local_xla//xla/stream_executor:stream_executor_h",
    ],
)

tf_cc_test(
    name = "device_compilation_cluster_signature_test",
    srcs = [
        "device_compilation_cluster_signature_test.cc",
    ],
    deps = [
        ":device_compilation_cluster_signature",
        ":flags",
        "//machina/compiler/tf2xla:common",
        "//machina/core:test",
        "//machina/core:test_main",
        "@local_xla//xla/client:client_library",
    ],
)

tf_cc_test(
    name = "device_compilation_profiler_test",
    srcs = ["device_compilation_profiler_test.cc"],
    tags = [
        "nomsan",  # TODO(b/284492454)
    ],
    deps = [
        ":device_compilation_profiler",
        ":xla_activity_proto_cc",
        "//machina/compiler/jit/tests:device_compiler_test_helper",
        "//machina/core:protos_all_cc",
        "@com_google_googletest//:gtest_main",
    ],
)

tf_cc_test(
    name = "device_executable_persistor_test",
    srcs = ["device_executable_persistor_test.cc"],
    tags = ["no_cuda_on_cpu_tap"],
    deps = [
        ":device_compiler_client",
        ":device_executable_persistor",
        ":pjrt_device_compiler_client",
        ":xla_compilation_cache_proto_cc",
        ":xla_cpu_device",
        ":xla_cpu_jit",
        ":xla_device_compiler_client",
        "//machina/cc:function_ops",
        "//machina/cc:math_ops",
        "//machina/cc:scope",
        "//machina/core:test",
        "//machina/core/platform:errors",
        "//machina/core/platform:status_matchers",
        "//machina/core/platform:statusor",
        "//machina/core/tfrt/common:create_pjrt_client_util",
        "//machina/core/tfrt/common:pjrt_util",
        "@com_google_googletest//:gtest_main",
        "@local_xla//xla/client:client_library",
        "@local_xla//xla/client:executable_build_options",
        "@local_xla//xla/client:local_client",
        "@local_xla//xla/pjrt:pjrt_client",
        "@local_xla//xla/pjrt/plugin/xla_cpu:cpu_client_options",
        "@local_xla//xla/pjrt/plugin/xla_cpu:xla_cpu_pjrt_client",
    ],
)

tf_cc_test(
    name = "device_compilation_cache_test",
    srcs = ["device_compilation_cache_test.cc"],
    deps = [
        ":device_compilation_cache",
        ":xla_compile_util",
        "//machina/compiler/tf2xla:xla_compiler",
        "//machina/core:test",
        "//machina/core/platform:errors",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest_main",
        "@local_xla//xla/hlo/builder:xla_computation",
        "@local_xla//xla/tsl/protobuf:error_codes_proto_impl_cc",
    ],
)

# TODO(b/261212343): Support running this test on CPU and OSS.
tf_cuda_cc_test(
    name = "device_compiler_test",
    srcs = ["device_compiler_test.cc"],
    tags = [
        "config-cuda-only",
        "no_oss",  # This test only runs with GPU.
        "requires-gpu-nvidia",
        "xla",
    ],
    deps = [
        ":device_compilation_cluster_signature",
        ":device_compiler",
        ":device_compiler_client",
        ":xla_compile_util",
        ":xla_device_compiler_client",
        ":xla_gpu_device",
        ":xla_gpu_jit",
        "//machina/cc:function_ops",
        "//machina/cc:math_ops",
        "//machina/cc:scope",
        "//machina/compiler/jit/tests:device_compiler_test_helper",
        "//machina/compiler/tf2xla:xla_compiler",
        "//machina/core:framework",
        "//machina/core:framework_internal",
        "//machina/core:test",
        "//machina/core/framework:fake_input",
        "//machina/core/kernels:ops_testutil",
        "//machina/core/platform:errors",
        "//machina/core/platform:notification",
        "//machina/core/platform:status",
        "//machina/core/platform:status_matchers",
        "//machina/core/platform:statusor",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_googletest//:gtest_main",
        "@local_xla//xla/client:client_library",
        "@local_xla//xla/client:local_client",
        "@local_xla//xla/hlo/builder:xla_computation",
        "@local_xla//xla/stream_executor:platform_manager",
    ],
)

tf_cuda_cc_test(
    name = "device_context_test",
    srcs = ["device_context_test.cc"],
    tags = tf_cuda_tests_tags() + [
        "config-cuda-only",
        "no_oss",  # Temporarily disable OSS.
    ],
    deps = [
        ":flags",
        ":xla_device",
        ":xla_gpu_device",
        ":xla_gpu_jit",
        "//machina/compiler/tf2xla:xla_op_registry",
        "//machina/core:framework_internal",
        "//machina/core:test",
        "//machina/core/framework:tensor_testutil",
        "@com_google_googletest//:gtest_main",
    ],
)

tf_cuda_cc_test(
    name = "xla_compiler_options_util_test",
    srcs = ["xla_compiler_options_util_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":device_compiler",
        ":device_executable_persistor",
        ":flags",
        ":pjrt_base_device",
        ":pjrt_device_compiler_client",
        ":test_util",
        ":xla_device_compiler_client",
        ":xla_device_no_jit_rewrite_registration",
        ":xla_gpu_device",
        ":xla_gpu_jit",
        "//machina/compiler/tf2xla:layout_util",
        "//machina/compiler/tf2xla:xla_argument",
        "//machina/compiler/tf2xla:xla_compiler",
        "//machina/compiler/tf2xla:xla_op_registry",
        "//machina/core:framework",
        "//machina/core:framework_types_hdr",
        "//machina/core/platform:refcount",
        "//machina/core/tpu:tpu_defs",
        "@com_google_googletest//:gtest_main",
        "@local_xla//xla:shape_util",
        "@local_xla//xla:xla_data_proto_cc",
        "@local_xla//xla/client:client_library",
        "@local_xla//xla/client:local_client",
        "@local_xla//xla/pjrt:pjrt_client",
        "@local_xla//xla/stream_executor:device_memory_allocator",
        "@local_xla//xla/stream_executor:platform",
        "@local_xla//xla/stream_executor/host:host_platform_id",
        "@local_xla//xla/tsl/platform:statusor",
    ],
)

tf_cuda_cc_test(
    name = "xla_platform_info_test",
    srcs = ["xla_platform_info_test.cc"],
    tags = tf_cuda_tests_tags() + ["config-cuda-only"],
    deps = [
        ":flags_headers",
        ":test_util",
        ":xla_device_no_jit_rewrite_registration",
        ":xla_gpu_device",
        ":xla_gpu_jit",
        "//machina/compiler/tf2xla:layout_util",
        "//machina/core:framework",
        "//machina/core:framework_types_hdr",
        "//machina/core:lib_proto_parsing",
        "//machina/core:test",
        "//machina/core/platform:refcount",
        "//machina/core/platform:status_matchers",
        "//machina/core/platform:statusor",
        "//machina/core/protobuf:error_codes_proto_impl_cc",
        "//machina/core/tfrt/common:create_pjrt_client_util",
        "//machina/core/tfrt/common:pjrt_util",
        "//machina/core/tpu:tpu_defs",
        "@com_google_googletest//:gtest_main",
        "@local_xla//xla/pjrt/plugin/xla_cpu:cpu_client_options",
        "@local_xla//xla/pjrt/plugin/xla_cpu:xla_cpu_pjrt_client",
    ],
)

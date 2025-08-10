#include "third_party/absl/strings/str_cat.h"
#DTensor C++ runtime and libraries.

load("//machina:machina.bzl", "tf_cc_test")
load("//machina:machina.default.bzl", "tf_kernel_library")
load(
    "//machina/core/platform:build_config.bzl",
    "tf_dtensor_tpu_dependencies",
)
load("//machina/core/platform:rules_cc.bzl", "cc_library")

default_visibility = [
    "//smartass/brain/jax/embeddings:__pkg__",
    "//machina/dtensor:dtensor-internal",
    "//machina:__pkg__",
]

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = default_visibility,
    licenses = ["notice"],
)

cc_library(
    name = "constants",
    hdrs = ["constants.h"],
)

cc_library(
    name = "dstatus",
    hdrs = ["dstatus.h"],
    deps = [
        "//machina/core:lib",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:optional",
    ],
)

cc_library(
    name = "dtensor_utils",
    srcs = ["dtensor_utils.cc"],
    hdrs = ["dtensor_utils.h"],
    deps = [
        "//machina/core:lib",
        "@com_google_absl//absl/strings",
        "@local_xla//xla/tsl/util:env_var",
    ],
)

cc_library(
    name = "tensor_layout",
    srcs = ["tensor_layout.cc"],
    hdrs = ["tensor_layout.h"],
    deps = [
        ":dstatus",
        "//machina/core:core_cpu_base",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/dtensor/proto:layout_proto_cc",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@toolchain-project//mlir:IR",
    ],
)

cc_library(
    name = "small_constant_optimization",
    srcs = ["small_constant_optimization.cc"],
    hdrs = ["small_constant_optimization.h"],
    deps = [
        ":constants",
        ":tensor_layout",
        "//machina/c:c_api",
        "//machina/c:c_api_experimental",
        "//machina/c:tf_tensor_internal",
        "//machina/c/eager:c_api",
        "//machina/c/eager:c_api_experimental",
        "//machina/core:framework",
        "//machina/core:framework_lite",
        "//machina/core:protos_all_cc",
        "//machina/core/platform:tstring",
        "//machina/dtensor/proto:layout_proto_cc",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
    ],
)

cc_library(
    name = "tensor_with_layout",
    srcs = ["tensor_with_layout.cc"],
    hdrs = ["tensor_with_layout.h"],
    deps = [
        ":constants",
        ":tensor_layout",
        "//machina/c/eager:c_api",
        "//machina/core:framework",
        "//machina/core/platform:fingerprint",
        "@toolchain-project//toolchain:Support",
    ],
)

cc_library(
    name = "dtensor_operation",
    hdrs = ["dtensor_operation.h"],
    deps = [
        ":tensor_layout",
        "//machina/c/eager:c_api",
    ],
)

cc_library(
    name = "dtensor_device_util",
    srcs = ["dtensor_device_util.cc"],
    hdrs = ["dtensor_device_util.h"],
    deps = [
        ":constants",
        ":dstatus",
        ":dtensor_operation",
        ":dtensor_utils",
        ":small_constant_optimization",
        ":tensor_layout",
        ":tensor_with_layout",
        "//machina/c:safe_ptr",
        "//machina/c:tf_datatype",
        "//machina/c:tf_status_headers",
        "//machina/c:tf_status_helper",
        "//machina/c:tf_tensor",
        "//machina/c:tf_tensor_internal",
        "//machina/c/eager:c_api",
        "//machina/c/eager:c_api_experimental",
        "//machina/c/eager:c_api_internal",
        "//machina/c/eager:tfe_context_internal",
        "//machina/c/eager:tfe_tensorhandle_internal",
        "//machina/c/eager/parallel_device:parallel_device_lib",
        "//machina/core:core_cpu",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core/common_runtime/eager:context",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:IR",
        "@local_tsl//tsl/platform:refcount",
        "@local_xla//xla:status_macros",
        "@local_xla//xla/tsl/platform:logging",
        "@local_xla//xla/tsl/platform:status",
    ],
)

cc_library(
    name = "dtensor_ops",
    srcs = [
        "dtensor_ops.cc",
    ],
    deps = [
        "//machina/core:framework",
        "//machina/core:protos_all_cc",
    ],
    alwayslink = 1,
)

# These ops are created only by DTensor MLIR passes and never by users, so they don't need Python wrappers.
cc_library(
    name = "dtensor_meta_ops",
    srcs = [
        "dtensor_meta_ops.cc",
    ],
    deps = [
        ":tensor_layout",
        "//machina/core:framework",
    ],
    alwayslink = 1,
)

cc_library(
    name = "tpu_system_interface",
    srcs = ["tpu_system_interface.cc"],
    hdrs = ["tpu_system_interface.h"],
    deps = [
        "//machina/core:framework",
        "//machina/core:lib",
        "@com_google_absl//absl/time",
    ],
)

cc_library(
    name = "save_restore_util",
    srcs = ["save_restore_util.cc"],
    hdrs = ["save_restore_util.h"],
    deps = [
        ":dstatus",
        ":tensor_layout",
        "//machina/compiler/mlir/machina",
        "//machina/dtensor/mlir:value_utils",
        "@com_google_absl//absl/container:flat_hash_map",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Support",
    ],
)

cc_library(
    name = "dtensor_tpu_ops",
    srcs = ["dtensor_tpu_ops.cc"],
    deps = [
        "//machina/core:framework",
        "//machina/core:protos_all_cc",
    ],
    alwayslink = 1,
)

tf_kernel_library(
    name = "dtensor_tpu_kernels",
    srcs = [
        "dtensor_tpu_kernels.cc",
    ],
    tags = [
        "no_rocm",
        "tpu",
    ],  # Disable building of TPU kernels on non-TPU platforms.
    deps = [
        ":dstatus",
        ":tpu_system_interface",
        "//machina/c:tf_status_headers",
        "//machina/c:tf_status_helper",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core/tpu:tpu_configuration",
        "//machina/core/tpu/kernels:tpu_compilation_cache_interface",
        "//machina/core/tpu/kernels:tpu_compilation_cache_local_lookup",
        "//machina/core/tpu/kernels:tpu_compilation_cache_lookup",
        "//machina/core/tpu/kernels:tpu_configuration_ops",
        "//machina/core/tpu/kernels:tpu_embedding_engine_state_interface",
        "//machina/core/tpu/kernels:tpu_mesh_state_interface",
        "//machina/core/tpu/kernels:tpu_op_consts",
        "//machina/core/tpu/kernels:tpu_pod_state",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/time",
        "@local_xla//xla/stream_executor/tpu:c_api_decl",
        "@local_xla//xla/stream_executor/tpu:status_helper",
        "@local_xla//xla/stream_executor/tpu:tpu_api",
        "@local_xla//xla/stream_executor/tpu:tpu_executor_base",
        "@local_xla//xla/stream_executor/tpu:tpu_ops_c_api_hdrs",
        "@local_xla//xla/stream_executor/tpu:tpu_topology_external",
    ],
    alwayslink = 1,
)

cc_library(
    name = "dtensor_graph_to_mlir_pass",
    srcs = ["dtensor_graph_to_mlir_pass.cc"],
    hdrs = ["dtensor_graph_to_mlir_pass.h"],
    deps = [
        ":constants",
        ":dtensor_utils",
        "//machina/compiler/jit:flags_headers",
        "//machina/compiler/mlir/machina",
        "//machina/compiler/mlir/machina:convert_type",
        "//machina/compiler/mlir/machina:device_util",
        "//machina/compiler/mlir/machina:error_util",
        "//machina/compiler/mlir/tf2xla/api/v2:graph_to_tf_executor",
        "//machina/core:core_cpu",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core/common_runtime:device_set",
        "//machina/dtensor/mlir:dtensor_mlir_passes",
        "//machina/dtensor/mlir:tf_dtensor_dialect",
        "//machina/dtensor/mlir/dtensor_dialect:Dialect",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@toolchain-project//mlir:AllExtensions",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:RegisterAllExtensions",
        "@toolchain-project//mlir:Support",
        "@local_xla//xla:status_macros",
        "@local_xla//xla/tsl/platform:status",
    ],
)

cc_library(
    name = "default_parallel_executor_lib",
    deps = [":default_parallel_executor"],
)

cc_library(
    name = "default_parallel_executor",
    srcs = ["default_parallel_executor.cc"],
    deps = [
        ":parallel_executor_interface",
        "//machina/core/platform:logging",
        "@local_xla//xla/tsl/platform:statusor",
    ],
)

cc_library(
    name = "parallel_executor_interface",
    hdrs = ["parallel_executor.h"],
    deps = [
        ":tensor_layout",
        ":tensor_with_layout",
        "//machina/c/eager:c_api_internal",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:IR",
        "@local_xla//xla/pjrt:pjrt_future",
    ],
)

cc_library(
    name = "dtensor_device_cc",
    srcs = ["dtensor_device.cc"],
    hdrs = ["dtensor_device.h"],
    deps = [
        ":constants",
        ":default_parallel_executor_lib",
        ":dstatus",
        ":dtensor_device_util",
        ":dtensor_graph_to_mlir_pass",
        ":dtensor_meta_ops",
        ":dtensor_operation",
        ":dtensor_ops",
        ":dtensor_tpu_ops",
        ":dtensor_utils",
        ":parallel_executor_interface",
        ":small_constant_optimization",
        ":tensor_layout",
        ":tpu_system_interface",
        "//machina/c:c_api",
        "//machina/c:c_api_experimental",
        "//machina/c:tf_status_helper",
        "//machina/c:tf_tensor_internal",
        "//machina/c/eager:c_api",
        "//machina/c/eager:c_api_experimental",
        "//machina/c/eager:tfe_context_internal",
        "//machina/c/eager:tfe_op_attrs_internal",
        "//machina/c/eager:tfe_tensorhandle_internal",
        "//machina/c/eager/parallel_device:parallel_device_lib",
        "//machina/compiler/mlir/machina:mlir_roundtrip_flags",
        "//machina/compiler/mlir/tf2xla/api/v2:tf_executor_to_graph",
        "//machina/core:core_cpu",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core/common_runtime:device_set",
        "//machina/core/common_runtime/eager:context",
        "//machina/core/common_runtime/eager:eager_operation",
        "//machina/core/common_runtime/eager:tensor_handle",
        "//machina/core/profiler/lib:traceme",
        "//machina/dtensor/mlir:layout_parsing",
        "//machina/dtensor/mlir:op_utils",
        "//machina/dtensor/mlir:spmd_expander",
        "//machina/dtensor/proto:layout_proto_cc",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:IR",
        "@local_xla//xla:status_macros",
        "@local_xla//xla/stream_executor/tpu:c_api_decl",
        "@local_xla//xla/stream_executor/tpu:tpu_platform_interface",
        "@local_xla//xla/stream_executor/tpu:tpu_topology_external",
        "@local_xla//xla/tsl/platform:status",
        "@local_xla//xla/tsl/util:env_var",
    ] + tf_dtensor_tpu_dependencies(),
)

cc_library(
    name = "layout_to_xla_sharding",
    srcs = ["xla_spmd/layout_to_xla_sharding.cc"],
    hdrs = ["xla_spmd/layout_to_xla_sharding.h"],
    deps = [
        ":dstatus",
        ":tensor_layout",
        "//machina/core:portable_gif_internal",
        "//machina/core/platform:status",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/types:span",
        "@toolchain-project//toolchain:Support",
        "@local_xla//xla:status_macros",
        "@local_xla//xla:xla_data_proto_cc",
    ],
)

cc_library(
    name = "mesh_type",
    hdrs = ["mesh_type.h"],
    deps = [
        ":tensor_layout",
        "//machina/c:conversion_macros",
    ],
)

cc_library(
    name = "slice_util",
    srcs = ["slice_util.cc"],
    hdrs = ["slice_util.h"],
    deps = [
        ":tensor_layout",
        "//machina/core:lib",
        "@toolchain-project//mlir:IR",
    ],
)

tf_cc_test(
    name = "small_constant_optimization_test",
    srcs = ["small_constant_optimization_test.cc"],
    deps = [
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/dtensor/cc:small_constant_optimization",
        "//machina/dtensor/cc:tensor_layout",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)

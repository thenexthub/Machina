load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = [
        "//machina/compiler/mlir/tfrt:__subpackages__",
        "//machina/core/tfrt:__subpackages__",
    ],
)

cc_library(
    name = "parallelization",
    srcs = ["parallelization.cc"],
    hdrs = ["parallelization.h"],
    deps = [
        "//machina/compiler/mlir/machina",
        "//machina/compiler/mlir/machina:machina_analysis",
        "//machina/compiler/mlir/tfrt:constants",
        "//machina/compiler/mlir/tfrt:cost_analysis",
        "//machina/compiler/mlir/tfrt/ir/mlrt:mlrt_ops",
        "//machina/compiler/mlir/tfrt/ir/mlrt:tf_mlrt_ops",
        "//machina/core/tfrt/fallback:cost_recorder",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Pass",
        "@tf_runtime//:stream_analysis",
    ],
)

cc_library(
    name = "assign_op_key",
    srcs = ["assign_op_key.cc"],
    hdrs = ["assign_op_key.h"],
    deps = [
        ":util",
        "//machina/compiler/mlir/tfrt:constants",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Pass",
    ],
)

cc_library(
    name = "mlrt_device_constants",
    hdrs = ["mlrt_device_constants.h"],
)

cc_library(
    name = "util",
    srcs = ["util.cc"],
    hdrs = ["util.h"],
    deps = [
        "//machina/compiler/mlir/machina",
        "//machina/compiler/mlir/machina:machina_ops_a_m_inc_gen",
        "//machina/compiler/mlir/machina:machina_ops_n_z_inc_gen",
        "//machina/compiler/mlir/machina/ir/host_runtime:machina_tfrt_ops_inc_gen",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:IR",
    ],
)

cc_library(
    name = "tf_to_mlrt",
    srcs = ["tf_to_mlrt.cc"],
    hdrs = ["tf_to_mlrt.h"],
    deps = [
        ":execute_op_registry",
        ":mlrt_device_constants",
        ":tpu_conversion_patterns",
        ":util",
        "//machina/compiler/mlir/machina",
        "//machina/compiler/mlir/machina:convert_tensor",
        "//machina/compiler/mlir/machina:translate_utils",
        "//machina/compiler/mlir/machina/ir/host_runtime:machina_tfrt_ops_inc_gen",
        "//machina/compiler/mlir/tfrt:constants",
        "//machina/compiler/mlir/tfrt:tfrt_pipeline_options",
        "//machina/compiler/mlir/tfrt:transform_utils",
        "//machina/compiler/mlir/tfrt/ir/mlrt:mlrt_ops",
        "//machina/compiler/mlir/tfrt/ir/mlrt:tf_mlrt_ops",
        "//machina/compiler/mlir/tfrt/ir/mlrt:tf_mlrt_tpu_ops",
        "//machina/core:framework",
        "//machina/core/platform:status",
        "//machina/core/tfrt/fallback:fallback_state",
        "//machina/core/tfrt/fallback:op_kernel_runner_cache",
        "@com_google_absl//absl/log",
        "@com_google_protobuf//:protobuf_headers",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:FuncTransforms",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:Support",
        "@toolchain-project//mlir:TransformUtils",
        "@local_xla//xla/tsl/platform:status",
    ],
)

cc_library(
    name = "passes",
    srcs = ["passes.cc"],
    hdrs = ["passes.h"],
    deps = [
        ":assign_op_key",
        ":async_while",
        ":fuse_mlrt_ops",
        ":ifrt_set_tpu_host_allocator",
        ":parallelization",
        ":rewrite_ifrt_load_variable",
        ":tf_to_mlrt",
        ":while_to_map_fn",
        "//machina/compiler/mlir/tfrt:tfrt_pipeline_options",
        "//machina/core/tfrt/fallback:cost_recorder",
        "//machina/core/tfrt/fallback:fallback_state",
        "@com_google_absl//absl/log:check",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:Transforms",
    ],
)

cc_library(
    name = "execute_op_registry",
    hdrs = ["execute_op_registry.h"],
    deps = [
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Support",
    ],
)

cc_library(
    name = "tpu_conversion_patterns",
    srcs = ["tpu_conversion_patterns.cc"],
    hdrs = ["tpu_conversion_patterns.h"],
    deps = [
        ":execute_op_registry",
        ":mlrt_device_constants",
        "//machina/compiler/mlir/machina",
        "//machina/compiler/mlir/tfrt:tfrt_pipeline_options",
        "//machina/compiler/mlir/tfrt:transform_utils",
        "//machina/compiler/mlir/tfrt/ir/mlrt:mlrt_ops",
        "//machina/compiler/mlir/tfrt/ir/mlrt:tf_mlrt_ops",
        "//machina/compiler/mlir/tfrt/ir/mlrt:tf_mlrt_tpu_ops",
        "@com_google_absl//absl/log:check",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Support",
        "@toolchain-project//mlir:TransformUtils",
    ],
)

cc_library(
    name = "fuse_mlrt_ops",
    srcs = ["fuse_mlrt_ops.cc"],
    hdrs = ["fuse_mlrt_ops.h"],
    deps = [
        "//machina/compiler/mlir/tfrt/ir/mlrt:mlrt_ops",
        "//machina/compiler/mlir/tfrt/ir/mlrt:tf_mlrt_ops",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:Pass",
    ],
)

cc_library(
    name = "import_model",
    srcs = ["import_model.cc"],
    hdrs = ["import_model.h"],
    deps = [
        ":assign_op_key",
        ":passes",
        ":while_to_map_fn",
        "//machina/compiler/mlir/machina:dump_mlir_util",
        "//machina/compiler/mlir/machina:error_util",
        "//machina/compiler/mlir/tf2xla/api/v2:tf_executor_to_graph",
        "//machina/compiler/mlir/tfrt:export",
        "//machina/compiler/mlir/tfrt:import_model",
        "//machina/compiler/mlir/tfrt:tf_to_tfrt",
        "//machina/compiler/mlir/tfrt:tfrt_compile_options",
        "//machina/compiler/mlir/tfrt:tfrt_pipeline_options",
        "//machina/compiler/mlir/tfrt/translate/mlrt:mlir_to_bytecode",
        "//machina/core:protos_all_cc",
        "//machina/core/platform:status",
        "//machina/core/platform:statusor",
        "//machina/core/tfrt/fallback:cost_recorder",
        "//machina/core/tfrt/fallback:fallback_state",
        "//machina/core/tfrt/mlrt/attribute",
        "//machina/core/tfrt/mlrt/bytecode",
        "//machina/core/tfrt/runtime",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:Support",
        "@toolchain-project//mlir:Transforms",
        "@local_xla//xla/tsl/platform:errors",
    ],
)

cc_library(
    name = "while_to_map_fn",
    srcs = ["while_to_map_fn.cc"],
    hdrs = ["while_to_map_fn.h"],
    deps = [
        "//machina/compiler/mlir/machina",
        "//machina/compiler/mlir/tfrt/ir/mlrt:mlrt_ops",
        "//machina/compiler/mlir/tfrt/ir/mlrt:tf_mlrt_ops",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:Support",
        "@toolchain-project//mlir:TransformUtils",
    ],
)

cc_library(
    name = "async_while",
    srcs = ["async_while.cc"],
    hdrs = ["async_while.h"],
    deps = [
        "//machina/compiler/mlir/machina",
        "//machina/compiler/mlir/machina:machina_analysis",
        "//machina/compiler/mlir/tfrt/ir/mlrt:mlrt_ops",
        "//machina/compiler/mlir/tfrt/ir/mlrt:tf_mlrt_ops",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:SideEffectInterfaces",
        "@toolchain-project//mlir:Support",
        "@toolchain-project//mlir:TransformUtils",
    ],
)

cc_library(
    name = "rewrite_ifrt_load_variable",
    srcs = ["rewrite_ifrt_load_variable.cc"],
    hdrs = ["rewrite_ifrt_load_variable.h"],
    deps = [
        "//machina/compiler/mlir/machina",
        "//machina/compiler/mlir/machina/ir/host_runtime:machina_tfrt_ops",
        "//machina/compiler/mlir/tfrt/ir/mlrt:mlrt_ops",
        "//machina/compiler/mlir/tfrt/ir/mlrt:tf_mlrt_ops",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:Support",
        "@toolchain-project//mlir:TransformUtils",
    ],
)

cc_library(
    name = "ifrt_set_tpu_host_allocator",
    srcs = ["ifrt_set_tpu_host_allocator.cc"],
    hdrs = ["ifrt_set_tpu_host_allocator.h"],
    deps = [
        ":mlrt_device_constants",
        "//machina/compiler/mlir/machina",
        "//machina/compiler/mlir/machina/ir/host_runtime:machina_tfrt_ops",
        "//machina/compiler/mlir/tfrt/ir/mlrt:mlrt_ops",
        "//machina/compiler/mlir/tfrt/ir/mlrt:tf_mlrt_ops",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:Support",
        "@toolchain-project//mlir:TransformUtils",
    ],
)

# Contains graph rewrites for TPU runtimes and optimizations.

load(
    "//machina:machina.bzl",
    "tf_cc_test",
)
load(
    "//machina/core/platform:build_config_root.bzl",
    "if_static",
)
load("//machina/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = [
        "//machina/compiler:__subpackages__",
        "//machina/core/tpu:__subpackages__",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "tpu_rewrite_pass_registration",
    srcs = ["tpu_rewrite_pass_registration.cc"],
    deps = [
        ":combine_tpu_embedding_load_retrieve_pass",
        ":distributed_tpu_configuration_rewrite_pass",
        ":distributed_tpu_rewrite_pass",
        ":encapsulate_tpu_computations_pass",
        ":tpu_embedding_software_deduplication_rewrite_pass",
        ":update_tpu_embedding_ops_passes",
        ":variable_merger_pass",
        "//machina/core:core_cpu",
    ],
    alwayslink = 1,
)

cc_library(
    name = "distributed_tpu_configuration_rewrite_pass",
    srcs = [
        "distributed_tpu_configuration_rewrite_pass.cc",
    ],
    hdrs = [
        "distributed_tpu_configuration_rewrite_pass.h",
    ],
    deps = [
        ":distributed_tpu_rewrite_helpers",
        "//machina/core:core_cpu_lib",
        "//machina/core:framework",
        "//machina/core:framework_internal",
        "//machina/core:lib",
        "//machina/core/framework:node_def_proto_cc",
        "//machina/core/framework:types_proto_cc",
        "//machina/core/tpu:tpu_init_mode",
        "//machina/core/tpu/kernels:tpu_compile_op_options",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@local_xla//xla:status_macros",
        "@local_xla//xla/tsl/platform:logging",
    ],
)

cc_library(
    name = "distributed_tpu_rewrite_helpers",
    srcs = ["distributed_tpu_rewrite_helpers.cc"],
    hdrs = ["distributed_tpu_rewrite_helpers.h"],
    deps = [
        "//machina/core:core_cpu_base",
        "//machina/core:framework",
        "//machina/core:portable_gif_internal",
        "//machina/core:protos_all_cc",
        "//machina/core/common_runtime:device_set",
        "//machina/core/platform:errors",
        "//machina/core/platform:status",
        "//machina/core/platform:str_util",
        "//machina/core/platform:strcat",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@local_xla//xla:status_macros",
        "@local_xla//xla/tsl/platform:errors",
    ],
)

cc_library(
    name = "variable_merger_pass",
    srcs = ["variable_merger_pass.cc"],
    hdrs = ["variable_merger_pass.h"],
    deps = [
        "//machina/core:framework",
        "//machina/core:graph",
        "//machina/core:lib",
        "//machina/core/common_runtime:optimization_registry",
        "//machina/core/framework:node_def_proto_cc",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@local_xla//xla:status_macros",
    ],
)

cc_library(
    name = "encapsulate_tpu_computations_pass",
    srcs = [
        "encapsulate_tpu_computations_pass.cc",
    ],
    hdrs = [
        "encapsulate_tpu_computations_pass.h",
    ],
    deps = [
        "//machina/compiler/jit:compilation_passes",
        "//machina/compiler/jit:encapsulate_util",
        "//machina/compiler/jit:xla_cluster_util",
        "//machina/compiler/tf2xla:side_effect_util",
        "//machina/compiler/tf2xla:tf2xla_util",
        "//machina/core:core_cpu_base",
        "//machina/core:framework",
        "//machina/core:graph",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core:session_options",
        "//machina/core/common_runtime:function_body",
        "//machina/core/common_runtime:function_utils",
        "//machina/core/config:flag_defs",
        "//machina/core/tpu:tpu_compile_interface",
        "//machina/core/tpu:tpu_defs",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@local_xla//xla:status_macros",
        "@local_xla//xla/tsl/platform:logging",
        "@local_xla//xla/tsl/platform:status",
        "@local_xla//xla/tsl/util:env_var",
    ] + if_static(
        [
            "//machina/core/common_runtime:function",
            "//machina/core/common_runtime:optimization_registry",
        ],
        [],
    ),
)

tf_cc_test(
    name = "encapsulate_tpu_computations_pass_test",
    srcs = ["encapsulate_tpu_computations_pass_test.cc"],
    deps = [
        ":encapsulate_tpu_computations_pass",
        "//machina/cc:cc_ops",
        "//machina/cc:cc_ops_internal",
        "//machina/core:framework",
        "//machina/core:framework_internal",
        "//machina/core:graph",
        "//machina/core:lib_internal",
        "//machina/core:ops",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core:testlib",
        "//machina/core/common_runtime:optimization_registry",
        "//machina/core/config:flag_defs",
        "//machina/core/framework:types_proto_cc",
    ],
)

cc_library(
    name = "distributed_tpu_rewrite_pass_internal",
    srcs = ["distributed_tpu_rewrite_pass_internal.cc"],
    hdrs = ["distributed_tpu_rewrite_pass_internal.h"],
    deps = ["@com_google_absl//absl/random"],
)

cc_library(
    name = "distributed_tpu_rewrite_pass",
    srcs = [
        "distributed_tpu_rewrite_pass.cc",
    ],
    hdrs = [
        "distributed_tpu_rewrite_pass.h",
    ],
    deps = [
        ":cond_builder",
        ":distributed_tpu_rewrite_helpers",
        ":distributed_tpu_rewrite_pass_internal",
        ":host_training_loop_optimization_util",
        ":incomplete_nodedef_builder",
        "//machina/compiler/jit:encapsulate_util",
        "//machina/compiler/jit:shape_inference",
        "//machina/compiler/mlir/machina:xla_sharding_util",
        "//machina/compiler/tf2xla:common",
        "//machina/compiler/tf2xla:resource_operation_table",
        "//machina/compiler/tf2xla:sharding_util",
        "//machina/compiler/tf2xla:side_effect_util",
        "//machina/compiler/tf2xla:tf2xla_util",
        "//machina/compiler/tf2xla:xla_compiler",
        "//machina/core:framework",
        "//machina/core:graph",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core:session_options",
        "//machina/core/common_runtime:device_propagation",
        "//machina/core/platform:error_payloads",
        "//machina/core/protobuf/tpu:compile_metadata_proto_cc",
        "//machina/core/protobuf/tpu:dynamic_padding_proto_cc",
        "//machina/core/protobuf/tpu:topology_proto_cc",
        "//machina/core/tpu:tpu_compile_interface",
        "//machina/core/tpu:tpu_defs",
        "//machina/core/tpu:tpu_fingerprint_utils",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@local_xla//xla:array2d",
        "@local_xla//xla:array3d",
        "@local_xla//xla:array4d",
        "@local_xla//xla:shape_util",
        "@local_xla//xla:status_macros",
        "@local_xla//xla:xla_data_proto_cc",
        "@local_xla//xla:xla_proto_cc",
        "@local_xla//xla/hlo/builder:sharding_builder",
        "@local_xla//xla/service:computation_placer",
        "@local_xla//xla/stream_executor/tpu:c_api_decl",
        "@local_xla//xla/stream_executor/tpu:tpu_api",
        "@local_xla//xla/stream_executor/tpu:tpu_ops_c_api_hdrs",
        "@local_xla//xla/stream_executor/tpu:tpu_platform_interface",
        "@local_xla//xla/stream_executor/tpu:tpu_topology_external",
        "@local_xla//xla/tsl/platform:logging",
    ] + if_static(
        [
            "//machina/core/common_runtime:function",
            "//machina/core/common_runtime:graph_constructor",
            "//machina/core/common_runtime:lower_function_call_op",
            "//machina/core/common_runtime:lower_functional_ops",
            "//machina/core/common_runtime:lower_if_op",
            "//machina/core/common_runtime:lower_while_op",
            "//machina/core/common_runtime:optimization_registry",
        ],
        [
            "//machina/core/common_runtime:core_cpu_base_no_ops",
        ],
    ),
)

cc_library(
    name = "incomplete_nodedef_builder",
    srcs = ["incomplete_nodedef_builder.cc"],
    hdrs = ["incomplete_nodedef_builder.h"],
    deps = [
        "//machina/core:core_cpu_lib",
        "//machina/core:framework",
        "//machina/core:framework_internal",
        "//machina/core:lib",
        "//machina/core/framework:node_def_proto_cc",
        "//machina/core/framework:types_proto_cc",
        "@com_google_absl//absl/status",
    ],
)

cc_library(
    name = "cond_builder",
    srcs = ["cond_builder.cc"],
    hdrs = ["cond_builder.h"],
    deps = [
        ":incomplete_nodedef_builder",
        "//machina/core:core_cpu_lib",
        "//machina/core:framework_internal",
        "//machina/core:lib",
        "//machina/core/framework:types_proto_cc",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@local_xla//xla/tsl/platform:status",
    ],
)

cc_library(
    name = "host_training_loop_optimization_util",
    srcs = [
        "host_training_loop_optimization_util.cc",
    ],
    hdrs = [
        "host_training_loop_optimization_util.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":distributed_tpu_rewrite_pass_internal",
        "//machina/compiler/tf2xla:functionalize_control_flow_util",
        "//machina/compiler/tf2xla:tf2xla_util",
        "//machina/core:core_cpu_base",
        "//machina/core:framework",
        "//machina/core:framework_internal",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core/framework:node_def_proto_cc",
        "//machina/core/framework:node_def_util",
        "//machina/core/framework:types_proto_cc",
        "//machina/core/protobuf/tpu:compile_metadata_proto_cc",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:node_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:tstring",
        "@local_xla//xla:status_macros",
        "@local_xla//xla/tsl/platform:logging",
    ],
)

cc_library(
    name = "tpu_embedding_rewrite_pass_utils",
    srcs = ["tpu_embedding_rewrite_pass_utils.cc"],
    hdrs = ["tpu_embedding_rewrite_pass_utils.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//machina/core:core_cpu",
        "//machina/core:core_cpu_internal",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core:protos_all_cc",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library(
    name = "combine_tpu_embedding_load_retrieve_pass",
    srcs = ["combine_tpu_embedding_load_retrieve_pass.cc"],
    hdrs = ["combine_tpu_embedding_load_retrieve_pass.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":tpu_embedding_rewrite_pass_utils",
        "//machina/core:core_cpu",
        "//machina/core:core_cpu_internal",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core:protos_all_cc",
        "//machina/core/protobuf/tpu:optimization_parameters_proto_cc",
        "//machina/core/protobuf/tpu:tpu_embedding_configuration_proto_cc",
        "//machina/core/tpu:tpu_embedding_optimization_parameters_utils",
        "//machina/core/tpu/ops:tpu_embedding_ops",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@local_xla//xla:status_macros",
        "@local_xla//xla/stream_executor/tpu:tpu_ops_c_api_hdrs",
        "@local_xla//xla/tsl/platform:logging",
    ],
)

cc_library(
    name = "configure_tpu_embedding_rewrite_pass",
    srcs = [
        "configure_tpu_embedding_rewrite_pass.cc",
    ],
    hdrs = [
        "configure_tpu_embedding_rewrite_pass.h",
    ],
    deps = [
        ":distributed_tpu_rewrite_helpers",
        "//machina/core:core_cpu_lib",
        "//machina/core:framework",
        "//machina/core:framework_internal",
        "//machina/core:lib",
        "//machina/core/protobuf/tpu:tpu_embedding_configuration_proto_cc",
        "//machina/core/tpu:tpu_embedding_configuration_proto_rewrite",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@local_xla//xla:status_macros",
    ],
)

cc_library(
    name = "configure_tpu_embedding_rewrite_registration",
    srcs = ["configure_tpu_embedding_rewrite_registration.cc"],
    deps = [
        ":configure_tpu_embedding_rewrite_pass",
        "//machina/core:core_cpu",
    ],
    alwayslink = 1,
)

cc_library(
    name = "tpu_embedding_software_deduplication_rewrite_pass",
    srcs = ["tpu_embedding_software_deduplication_rewrite_pass.cc"],
    hdrs = ["tpu_embedding_software_deduplication_rewrite_pass.h"],
    deps = [
        ":tpu_embedding_rewrite_pass_utils",
        "//machina/core:core_cpu",
        "//machina/core:core_cpu_internal",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core:protos_all_cc",
        "//machina/core/protobuf/tpu:tpu_embedding_configuration_proto_cc",
        "//machina/core/tpu:tpu_embedding_configuration_utils",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@local_xla//xla:status_macros",
    ],
)

cc_library(
    name = "update_tpu_embedding_ops_passes",
    srcs = [
        "update_tpu_embedding_ops_passes.cc",
    ],
    hdrs = [
        "update_tpu_embedding_ops_passes.h",
    ],
    deps = [
        "//machina/compiler/tf2xla:side_effect_util",
        "//machina/core:core_cpu_lib",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core/framework:function_proto_cc",
        "//machina/core/framework:node_def_proto_cc",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@local_xla//xla:status_macros",
    ],
)

load(
    "//machina:machina.bzl",
    "if_not_mobile",
    "tf_cc_test",
)
load(
    "//machina/core/platform:build_config.bzl",
    "tf_protos_all",
)
load("//machina/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = [
        "//machina:internal",
    ],
    licenses = ["notice"],
)

# Export files for use on Android.
exports_files([
    "captured_function.cc",
    "captured_function.h",
    "compression_utils.cc",
    "compression_utils.h",
    "dataset_utils.cc",
    "dataset_utils.h",
    "finalization_utils.cc",
    "finalization_utils.h",
    "flat_map_utils.cc",
    "flat_map_utils.h",
    "global_shuffle_utils.cc",
    "global_shuffle_utils.h",
    "metric_utils.cc",
    "metric_utils.h",
    "name_utils.cc",
    "name_utils.h",
    "rewrite_utils.cc",
    "rewrite_utils.h",
    "root_dataset.cc",
    "root_dataset.h",
    "serialization_utils.cc",
    "serialization_utils.h",
    "split_utils.cc",
    "split_utils.h",
    "stats_utils.cc",
    "stats_utils.h",
    "tf_data_memory_logger.cc",
    "tf_data_memory_logger.h",
    "tfdataz_metrics.h",
    "tfdataz_metrics.cc",
    "unbounded_thread_pool.cc",
    "unbounded_thread_pool.h",
    "utils.cc",
    "utils.h",
])

cc_library(
    name = "captured_function",
    srcs = ["captured_function.cc"],
    hdrs = ["captured_function.h"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    visibility = ["//visibility:public"],
    deps = [
        ":dataset_utils",
        ":stats_utils",
        "//machina/core:core_cpu_internal",
        "//machina/core:framework",
        "//machina/core:framework_internal",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core:protos_all_cc",
        "//machina/core/profiler/lib:traceme",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/time",
    ] + if_not_mobile([
        "//machina/core/grappler:grappler_item",
        "//machina/core/grappler/optimizers:meta_optimizer",
    ]),
)

cc_library(
    name = "compression_utils",
    srcs = ["compression_utils.cc"],
    hdrs = ["compression_utils.h"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    visibility = ["//machina:internal"],
    deps = [
        "//machina/core:core_cpu",
        "//machina/core:core_cpu_internal",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core:protos_all_cc",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
    ],
)

tf_cc_test(
    name = "compression_utils_test",
    srcs = ["compression_utils_test.cc"],
    # copybara:uncomment extra_copts = ["-Wthread-safety-analysis"],
    tags = [
        "requires-mem:24g",
    ],
    deps = [
        ":compression_utils",
        ":dataset_test_base",
        "//machina/core:framework",
        "//machina/core:lib_internal",
        "//machina/core:protos_all_cc",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core:testlib",
        "@com_google_googletest//:gtest",
        "@local_xla//xla/tsl/platform:status_matchers",
        "@local_xla//xla/tsl/protobuf:error_codes_proto_impl_cc",
    ],
)

cc_library(
    name = "dataset_test_base",
    testonly = 1,
    srcs = ["dataset_test_base.cc"],
    hdrs = ["dataset_test_base.h"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    visibility = ["//visibility:public"],
    deps = [
        ":dataset_utils",
        ":name_utils",
        ":serialization_utils",
        ":split_utils",
        "//machina/core:core_cpu",
        "//machina/core:core_cpu_internal",
        "//machina/core:framework",
        "//machina/core:framework_internal",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core:protos_all_cc",
        "//machina/core:test",
        "//machina/core:testlib",
        "//machina/core/framework:tensor_testutil",
        "//machina/core/kernels:function_ops",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@eigen_archive//:eigen3",
        "@local_xla//xla/tsl/framework/fixedpoint",
    ],
)

cc_library(
    name = "dataset_utils",
    srcs = ["dataset_utils.cc"],
    hdrs = ["dataset_utils.h"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    visibility = ["//visibility:public"],
    deps = [
        "//machina/core:core_cpu_internal",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core:protos_all_cc",
        "//machina/core/platform:regexp",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "dataset_utils_test",
    size = "small",
    srcs = ["dataset_utils_test.cc"],
    # copybara:uncomment extra_copts = ["-Wthread-safety-analysis"],
    deps = [
        ":compression_utils",
        ":dataset_test_base",
        ":dataset_utils",
        ":serialization_utils",
        ":test_utils",
        "//machina/cc:cc_ops",
        "//machina/core:framework",
        "//machina/core:protos_all_cc",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core/platform:str_util",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@local_xla//xla/tsl/platform:status_matchers",
        "@local_xla//xla/tsl/protobuf:error_codes_proto_impl_cc",
        "@local_xla//xla/tsl/util:determinism_test_util",
    ],
)

cc_library(
    name = "flat_map_utils",
    srcs = ["flat_map_utils.cc"],
    hdrs = ["flat_map_utils.h"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        ":captured_function",
        "//machina/core:core_cpu_base",
        "//machina/core:framework",
        "//machina/core/framework:types_proto_cc",
        "//machina/core/kernels/data:iterator_ops",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@local_tsl//tsl/platform:refcount",
        "@local_xla//xla/tsl/platform:env",
        "@local_xla//xla/tsl/platform:errors",
        "@local_xla//xla/tsl/platform:statusor",
    ],
)

cc_library(
    name = "global_shuffle_utils",
    srcs = ["global_shuffle_utils.cc"],
    hdrs = ["global_shuffle_utils.h"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        "//machina/core:framework",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@local_xla//xla/tsl/platform:errors",
        "@local_xla//xla/tsl/platform:statusor",
    ],
)

cc_library(
    name = "hash_utils",
    srcs = ["hash_utils.cc"],
    hdrs = ["hash_utils.h"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        ":dataset_utils",
        ":serialization_utils",
        "//machina/core:core_cpu_internal",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core:protos_all_cc",
        "//machina/core/platform:regexp",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "hash_utils_test",
    size = "small",
    srcs = ["hash_utils_test.cc"],
    # copybara:uncomment extra_copts = ["-Wthread-safety-analysis"],
    deps = [
        ":hash_utils",
        "//machina/cc:cc_ops",
        "//machina/core:framework",
        "//machina/core:protos_all_cc",
        "//machina/core:test",
        "//machina/core:test_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
        "@local_xla//xla/tsl/platform:test_main",
        "@local_xla//xla/tsl/protobuf:error_codes_proto_impl_cc",
    ],
)

cc_library(
    name = "tf_data_memory_logger",
    srcs = ["tf_data_memory_logger.cc"],
    hdrs = ["tf_data_memory_logger.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":tfdataz_metrics",
        "//machina/core/framework:model_proto_cc",
        "//machina/core/platform:env",
        "//machina/core/platform:numbers",
        "//machina/core/platform:status",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/container:flat_hash_set",
        "@local_xla//xla/tsl/platform:logging",
    ],
)

cc_library(
    name = "metric_utils",
    srcs = ["metric_utils.cc"],
    hdrs = [
        "metric_utils.h",
    ],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    visibility = ["//visibility:public"],
    deps = [
        ":tfdataz_metrics",
        "//machina/core:framework",
        "//machina/core/data:utils",
        "//machina/core/platform:env",
        "//machina/core/platform:mutex",
        "//machina/core/platform:thread_annotations",
        "@com_google_absl//absl/time",
    ],
)

tf_cc_test(
    name = "metric_utils_test",
    size = "small",
    srcs = ["metric_utils_test.cc"],
    # copybara:uncomment extra_copts = ["-Wthread-safety-analysis"],
    deps = [
        ":metric_utils",
        "//machina/core:framework",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core/lib/monitoring:cell_reader",
        "//machina/core/lib/monitoring:test_utils",
        "//machina/core/platform:env",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/time",
    ],
)

cc_library(
    name = "tfdataz_metrics",
    srcs = ["tfdataz_metrics.cc"],
    hdrs = ["tfdataz_metrics.h"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    visibility = ["//visibility:public"],
    deps = [
        "//machina/core:framework",
        "//machina/core:framework_internal",
        "//machina/core:lib",
        "//machina/core/platform:env",
        "//machina/core/platform:mutex",
        "//machina/core/platform:thread_annotations",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/time",
    ],
)

tf_cc_test(
    name = "tfdataz_metrics_test",
    size = "small",
    srcs = ["tfdataz_metrics_test.cc"],
    # copybara:uncomment extra_copts = ["-Wthread-safety-analysis"],
    deps = [
        ":tfdataz_metrics",
        "//machina/core:framework",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core/platform:env",
        "//machina/core/util:fake_clock_env",
        "@com_google_absl//absl/time",
    ],
)

cc_library(
    name = "name_utils",
    srcs = ["name_utils.cc"],
    hdrs = ["name_utils.h"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    visibility = ["//visibility:public"],
    deps = [
        "//machina/core:lib",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "name_utils_test",
    size = "small",
    srcs = ["name_utils_test.cc"],
    # copybara:uncomment extra_copts = ["-Wthread-safety-analysis"],
    deps = [
        ":name_utils",
        "//machina/core:framework",
        "//machina/core:test",
        "//machina/core:test_main",
    ],
)

cc_library(
    name = "rewrite_utils",
    srcs = ["rewrite_utils.cc"],
    hdrs = ["rewrite_utils.h"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        ":dataset_utils",
        ":hash_utils",
        ":serialization_utils",
        "//machina/core:core_cpu_internal",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core:protos_all_cc",
        "//machina/core/grappler:grappler_item",
        "//machina/core/grappler:grappler_item_builder",
        "//machina/core/grappler/clusters:virtual_cluster",
        "//machina/core/grappler/optimizers:custom_graph_optimizer_registry",
        "//machina/core/grappler/optimizers:meta_optimizer",
        "//machina/core/grappler/optimizers/data",
        "//machina/core/grappler/optimizers/data:function_utils",
        "//machina/core/grappler/optimizers/data:graph_utils",
        "//machina/core/platform",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "rewrite_utils_test",
    size = "small",
    srcs = ["rewrite_utils_test.cc"],
    # copybara:uncomment extra_copts = ["-Wthread-safety-analysis"],
    deps = [
        ":rewrite_utils",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core/framework:function_proto_cc",
        "//machina/core/framework:function_testlib",
        "//machina/core/framework:graph_proto_cc",
        "//machina/core/framework:tensor_testutil",
        "//machina/core/framework:types_proto_cc",
        "//machina/core/grappler:grappler_item",
        "//machina/core/ops",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "root_dataset",
    srcs = ["root_dataset.cc"],
    hdrs = ["root_dataset.h"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        ":dataset_utils",
        ":name_utils",
        ":rewrite_utils",
        "//machina/core:framework",
        "//machina/core:framework_internal",
        "//machina/core:lib_internal",
        "//machina/core:protos_all_cc",
        "//machina/core/platform:errors",
        "//machina/core/platform:platform_port",
        "//machina/core/platform:status",
        "//machina/core/platform:stringprintf",
        "@com_google_absl//absl/status",
    ],
)

cc_library(
    name = "serialization_utils",
    srcs = ["serialization_utils.cc"],
    hdrs = ["serialization_utils.h"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        ":compression_utils",
        ":dataset_utils",
        "//machina/core:core_cpu_base",
        "//machina/core:framework",
        "//machina/core:protos_all_cc",
        "//machina/core/lib/core:status",
        "//machina/core/platform:stringpiece",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@local_xla//xla/tsl/platform:statusor",
    ],
)

tf_cc_test(
    name = "serialization_utils_test",
    size = "small",
    srcs = ["serialization_utils_test.cc"],
    # copybara:uncomment extra_copts = ["-Wthread-safety-analysis"],
    deps = [
        ":dataset_test_base",
        ":dataset_utils",
        ":serialization_utils",
        ":test_utils",
        "//machina/cc:cc_ops",
        "//machina/core:framework",
        "//machina/core:protos_all_cc",
        "//machina/core:session_options",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core/common_runtime:device_factory",
        "//machina/core/framework:tensor_testutil",
        "//machina/core/platform:errors",
        "//machina/core/platform:str_util",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status:statusor",
        "@local_xla//xla/tsl/platform:statusor",
        "@local_xla//xla/tsl/protobuf:error_codes_proto_impl_cc",
    ],
)

cc_library(
    name = "split_utils",
    srcs = ["split_utils.cc"],
    hdrs = ["split_utils.h"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core/framework:types_proto_cc",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:mutex",
        "@local_tsl//tsl/platform:thread_annotations",
        "@local_xla//xla/tsl/platform:errors",
        "@local_xla//xla/tsl/platform:logging",
        "@local_xla//xla/tsl/platform:types",
    ],
)

tf_cc_test(
    name = "split_utils_test",
    size = "small",
    srcs = ["split_utils_test.cc"],
    # copybara:uncomment extra_copts = ["-Wthread-safety-analysis"],
    deps = [
        ":dataset_test_base",
        ":serialization_utils",
        ":split_utils",
        "//machina/core:framework",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core/data:dataset_utils",
        "//machina/core/framework:tensor_testutil",
    ],
)

cc_library(
    name = "snapshot_utils",
    srcs = ["snapshot_utils.cc"],
    hdrs = ["snapshot_utils.h"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        ":name_utils",
        "//machina/core:core_cpu_lib",
        "//machina/core:dataset_ops_op_lib",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core:protos_all_cc",
        "//machina/core/platform:coding",
        "//machina/core/platform:random",
        "//machina/core/profiler/lib:traceme",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@local_xla//xla/tsl/platform:errors",
        "@local_xla//xla/tsl/platform:status",
        "@local_xla//xla/tsl/platform:statusor",
    ],
)

tf_cc_test(
    name = "snapshot_utils_test",
    size = "small",
    srcs = ["snapshot_utils_test.cc"],
    # copybara:uncomment extra_copts = ["-Wthread-safety-analysis"],
    deps = [
        ":snapshot_utils",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core/data/service:test_util",
    ],
)

cc_library(
    name = "standalone",
    srcs = ["standalone.cc"],
    hdrs = ["standalone.h"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        ":dataset_utils",
        ":root_dataset",
        ":serialization_utils",
        ":tf_data_memory_logger",
        ":tfdataz_metrics",
        ":unbounded_thread_pool",
        "//machina/core:all_kernels",
        "//machina/core:core_cpu_internal",
        "//machina/core:framework",
        "//machina/core:framework_internal",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core:session_options",
        "//machina/core/framework:graph_proto_cc",
        "@com_google_absl//absl/memory",
        "@local_tsl//tsl/platform:refcount",
        "@local_xla//xla/tsl/platform:env",
        "@local_xla//xla/tsl/platform:errors",
        "@local_xla//xla/tsl/platform:status",
        "@local_xla//xla/tsl/platform:statusor",
    ],
)

tf_cc_test(
    name = "standalone_save_restore_test",
    srcs = ["standalone_save_restore_test.cc"],
    # copybara:uncomment extra_copts = ["-Wthread-safety-analysis"],
    deps = [
        ":standalone",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core/data/service:common_proto_cc",
        "//machina/core/data/service:test_util",
        "@local_xla//xla/tsl/platform:errors",
        "@local_xla//xla/tsl/platform:status",
        "@local_xla//xla/tsl/platform:status_matchers",
        "@local_xla//xla/tsl/platform:statusor",
        "@local_xla//xla/tsl/protobuf:protos_all_cc",
    ] + tf_protos_all(),
)

tf_cc_test(
    name = "standalone_test",
    srcs = ["standalone_test.cc"],
    # copybara:uncomment extra_copts = ["-Wthread-safety-analysis"],
    deps = [
        ":standalone",
        "//machina/core:framework",
        "//machina/core:test",
        "//machina/core:test_main",
    ] + tf_protos_all(),
)

cc_library(
    name = "stats_utils",
    srcs = ["stats_utils.cc"],
    hdrs = ["stats_utils.h"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        "//machina/core:lib",
        "@com_google_absl//absl/base:core_headers",
    ],
)

cc_library(
    name = "test_utils",
    srcs = ["test_utils.cc"],
    hdrs = ["test_utils.h"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        "//machina/core:core_cpu",
        "//machina/core:core_cpu_base",
        "//machina/core:framework",
        "//machina/core:protos_all_cc",
        "//machina/core/framework:function_proto_cc",
        "//machina/core/public:version",
        "@local_xla//xla/tsl/platform:env",
        "@local_xla//xla/tsl/platform:statusor",
    ],
)

cc_library(
    name = "unbounded_thread_pool",
    srcs = ["unbounded_thread_pool.cc"],
    hdrs = ["unbounded_thread_pool.h"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        "//machina/core:core_cpu_internal",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "@com_google_absl//absl/memory",
    ],
)

cc_library(
    name = "finalization_utils",
    srcs = ["finalization_utils.cc"],
    hdrs = ["finalization_utils.h"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        ":root_dataset",
        "//machina/core:framework",
        "//machina/core/platform:errors",
        "//machina/core/platform:refcount",
        "@com_google_absl//absl/status:statusor",
    ],
)

tf_cc_test(
    name = "unbounded_thread_pool_test",
    size = "small",
    srcs = ["unbounded_thread_pool_test.cc"],
    # copybara:uncomment extra_copts = ["-Wthread-safety-analysis"],
    deps = [
        ":unbounded_thread_pool",
        "//machina/core:lib_internal",
        "//machina/core:test",
        "//machina/core:test_main",
    ],
)

cc_library(
    name = "utils",
    srcs = ["utils.cc"],
    hdrs = ["utils.h"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        "//machina/core:framework",
        "//machina/core:protos_all_cc",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status:statusor",
    ],
)

tf_cc_test(
    name = "utils_test",
    srcs = ["utils_test.cc"],
    # copybara:uncomment extra_copts = ["-Wthread-safety-analysis"],
    deps = [
        ":utils",
        "//machina/core:framework",
        "@com_google_googletest//:gtest_main",
    ],
)

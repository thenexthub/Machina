# Description:
#   OpKernels for tf.data

load("//machina:machina.bzl", "tf_cc_test")

# Definitions are loaded separately so that copybara can pattern match (and modify) each definition.
load("//machina:machina.default.bzl", "filegroup", "tf_kernel_library")
load("//machina/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//machina:internal"],
    licenses = ["notice"],
)

tf_kernel_library(
    name = "batch_dataset_op",
    srcs = ["batch_dataset_op.cc"],
    hdrs = ["batch_dataset_op.h"],
    deps = [
        "//machina/core:dataset_ops_op_lib",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core/data:dataset_utils",
        "//machina/core/data:global_shuffle_utils",
        "//machina/core/data:name_utils",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:mutex",
    ],
)

tf_cc_test(
    name = "batch_dataset_op_test",
    size = "small",
    srcs = ["batch_dataset_op_test.cc"],
    deps = [
        ":batch_dataset_op",
        ":iterator_ops",
        ":range_dataset_op",
        "//machina/core",
        "//machina/core:core_cpu_internal",
        "//machina/core:dataset_ops_op_lib",
        "//machina/core:framework",
        "//machina/core:lib_internal",
        "//machina/core:session_options",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core:testlib",
        "//machina/core/common_runtime:type_inference",
        "//machina/core/data:dataset_test_base",
        "//machina/core/data:dataset_utils",
    ],
)

tf_kernel_library(
    name = "cache_dataset_ops",
    srcs = ["cache_dataset_ops.cc"],
    hdrs = ["cache_dataset_ops.h"],
    deps = [
        ":cache_ops",
        ":iterator_ops",
        "//machina/core:dataset_ops_op_lib",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core/data:global_shuffle_utils",
        "//machina/core/data:name_utils",
        "//machina/core/data:serialization_utils",
        "//machina/core/framework:dataset_options_proto_cc",
        "//machina/core/util/tensor_bundle",
        "//machina/core/util/tensor_bundle:naming",
        "@com_google_absl//absl/status",
    ],
)

tf_cc_test(
    name = "cache_dataset_ops_test",
    size = "small",
    srcs = ["cache_dataset_ops_test.cc"],
    deps = [
        ":cache_dataset_ops",
        ":iterator_ops",
        ":tensor_slice_dataset_op",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core:testlib",
        "//machina/core/data:dataset_test_base",
        "//machina/core/data:dataset_utils",
        "//machina/core/data:serialization_utils",
        "//machina/core/framework:dataset_options_proto_cc",
    ],
)

tf_kernel_library(
    name = "cache_ops",
    srcs = ["cache_ops.cc"],
    hdrs = ["cache_ops.h"],
    deps = [
        "//machina/core:core_cpu_internal",
        "//machina/core:framework",
        "//machina/core:functional_ops_op_lib",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core/data:dataset_utils",
    ],
)

tf_kernel_library(
    name = "concatenate_dataset_op",
    srcs = ["concatenate_dataset_op.cc"],
    hdrs = ["concatenate_dataset_op.h"],
    deps = [
        "//machina/core:dataset_ops_op_lib",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core/data:name_utils",
        "//machina/core/data:split_utils",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@local_tsl//tsl/platform:mutex",
    ],
)

tf_cc_test(
    name = "concatenate_dataset_op_test",
    size = "small",
    srcs = ["concatenate_dataset_op_test.cc"],
    deps = [
        ":concatenate_dataset_op",
        ":iterator_ops",
        ":tensor_slice_dataset_op",
        "//machina/core:core_cpu_internal",
        "//machina/core:dataset_ops_op_lib",
        "//machina/core:framework",
        "//machina/core:lib_internal",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core:testlib",
        "//machina/core/data:dataset_test_base",
        "//machina/core/data:dataset_utils",
    ],
)

tf_kernel_library(
    name = "dataset_ops",
    srcs = ["dataset_ops.cc"],
    hdrs = ["dataset_ops.h"],
    deps = [
        "//machina/core:core_cpu_internal",
        "//machina/core:dataset_ops_op_lib",
        "//machina/core:framework",
        "//machina/core:lib_internal",
        "//machina/core:protos_all_cc",
        "//machina/core/data:captured_function",
        "//machina/core/data:dataset_utils",
        "//machina/core/data:hash_utils",
        "//machina/core/data:serialization_utils",
        "//machina/core/grappler:graph_topology_view",
        "//machina/core/grappler/utils:traversal",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:str_format",
    ],
)

tf_kernel_library(
    name = "filter_dataset_op",
    srcs = ["filter_dataset_op.cc"],
    hdrs = ["filter_dataset_op.h"],
    deps = [
        "//machina/core:core_cpu_internal",
        "//machina/core:dataset_ops_op_lib",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core/data:captured_function",
        "//machina/core/data:dataset_utils",
        "//machina/core/data:name_utils",
        "//machina/core/data:stats_utils",
    ],
)

tf_cc_test(
    name = "filter_dataset_op_test",
    size = "small",
    srcs = ["filter_dataset_op_test.cc"],
    deps = [
        ":filter_dataset_op",
        ":iterator_ops",
        ":tensor_slice_dataset_op",
        "//machina/core:core_cpu_internal",
        "//machina/core:dataset_ops_op_lib",
        "//machina/core:framework",
        "//machina/core:lib_internal",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core:testlib",
        "//machina/core/data:dataset_test_base",
        "//machina/core/data:dataset_utils",
        "//machina/core/kernels:cwise_op",
        "//machina/core/kernels:function_ops",
        "//machina/core/kernels:unique_op",
    ],
)

tf_kernel_library(
    name = "finalize_dataset_op",
    srcs = ["finalize_dataset_op.cc"],
    hdrs = ["finalize_dataset_op.h"],
    deps = [
        ":model_dataset_op",
        ":optimize_dataset_op",
        "//machina/core:dataset_ops_op_lib",
        "//machina/core:framework",
        "//machina/core:framework_internal",
        "//machina/core:lib_internal",
        "//machina/core/data:dataset_utils",
        "//machina/core/data:name_utils",
        "//machina/core/framework:dataset_options_proto_cc",
        "//machina/core/kernels/data/experimental:threadpool_dataset_op",
        "//machina/core/profiler/lib:traceme",
        "//machina/core/protobuf:for_core_protos_cc",
    ],
)

tf_cc_test(
    name = "finalize_dataset_op_test",
    size = "small",
    srcs = ["finalize_dataset_op_test.cc"],
    deps = [
        ":finalize_dataset_op",
        ":options_dataset_op",
        ":prefetch_dataset_op",
        ":range_dataset_op",
        "//machina/core:core_cpu_internal",
        "//machina/core:dataset_ops_op_lib",
        "//machina/core:framework",
        "//machina/core:lib_internal",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core:testlib",
        "//machina/core/data:dataset_test_base",
        "//machina/core/data:dataset_utils",
    ],
)

tf_kernel_library(
    name = "fixed_length_record_dataset_op",
    srcs = ["fixed_length_record_dataset_op.cc"],
    hdrs = ["fixed_length_record_dataset_op.h"],
    deps = [
        "//machina/core:core_cpu_internal",
        "//machina/core:dataset_ops_op_lib",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core/data:name_utils",
        "//machina/core/data:utils",
    ],
)

tf_cc_test(
    name = "fixed_length_record_dataset_op_test",
    size = "small",
    srcs = ["fixed_length_record_dataset_op_test.cc"],
    deps = [
        ":fixed_length_record_dataset_op",
        ":iterator_ops",
        "//machina/core:core_cpu_internal",
        "//machina/core:dataset_ops_op_lib",
        "//machina/core:framework",
        "//machina/core:lib_internal",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core:testlib",
        "//machina/core/data:dataset_test_base",
        "//machina/core/data:dataset_utils",
    ],
)

tf_kernel_library(
    name = "flat_map_dataset_op",
    srcs = ["flat_map_dataset_op.cc"],
    hdrs = ["flat_map_dataset_op.h"],
    deps = [
        "//machina/core:core_cpu_internal",
        "//machina/core:dataset_ops_op_lib",
        "//machina/core:framework",
        "//machina/core:lib_internal",
        "//machina/core:protos_all_cc",
        "//machina/core/data:captured_function",
        "//machina/core/data:dataset_utils",
        "//machina/core/data:flat_map_utils",
        "//machina/core/data:name_utils",
        "//machina/core/data:serialization_utils",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:thread_annotations",
        "@local_xla//xla/tsl/platform:logging",
    ],
)

tf_cc_test(
    name = "flat_map_dataset_op_test",
    size = "small",
    srcs = ["flat_map_dataset_op_test.cc"],
    deps = [
        ":batch_dataset_op",
        ":flat_map_dataset_op",
        ":iterator_ops",
        ":tensor_slice_dataset_op",
        "//machina/core:core_cpu_internal",
        "//machina/core:dataset_ops_op_lib",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core:testlib",
        "//machina/core/data:captured_function",
        "//machina/core/data:dataset_test_base",
        "//machina/core/data:dataset_utils",
        "//machina/core/kernels:function_ops",
        "//machina/core/kernels:identity_op",
    ],
)

tf_kernel_library(
    name = "generator_dataset_op",
    srcs = ["generator_dataset_op.cc"],
    hdrs = ["generator_dataset_op.h"],
    deps = [
        "//machina/core:core_cpu_internal",
        "//machina/core:dataset_ops_op_lib",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core/data:captured_function",
        "//machina/core/data:dataset_utils",
        "//machina/core/data:name_utils",
    ],
)

tf_kernel_library(
    name = "get_options_op",
    srcs = ["get_options_op.cc"],
    hdrs = ["get_options_op.h"],
    deps = [
        "//machina/core:dataset_ops_op_lib",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core/data:name_utils",
        "//machina/core/framework:dataset_options_proto_cc",
        "//machina/core/profiler/lib:traceme",
        "//machina/core/protobuf:for_core_protos_cc",
        "@com_google_absl//absl/memory",
    ],
)

tf_cc_test(
    name = "get_options_op_test",
    size = "small",
    srcs = ["get_options_op_test.cc"],
    deps = [
        ":get_options_op",
        ":options_dataset_op",
        ":range_dataset_op",
        "//machina/core:core_cpu_internal",
        "//machina/core:dataset_ops_op_lib",
        "//machina/core:framework",
        "//machina/core:lib_internal",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core:testlib",
        "//machina/core/data:dataset_test_base",
        "//machina/core/data:dataset_utils",
    ],
)

tf_kernel_library(
    name = "interleave_dataset_op",
    srcs = ["interleave_dataset_op.cc"],
    hdrs = ["interleave_dataset_op.h"],
    deps = [
        "//machina/core:core_cpu_internal",
        "//machina/core:dataset_ops_op_lib",
        "//machina/core:framework",
        "//machina/core:framework_internal",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core/data:captured_function",
        "//machina/core/data:dataset_utils",
        "//machina/core/data:name_utils",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
    ],
)

tf_cc_test(
    name = "interleave_dataset_op_test",
    size = "small",
    srcs = ["interleave_dataset_op_test.cc"],
    deps = [
        ":interleave_dataset_op",
        ":iterator_ops",
        ":tensor_slice_dataset_op",
        "//machina/core:core_cpu_internal",
        "//machina/core:dataset_ops_op_lib",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core:testlib",
        "//machina/core/data:captured_function",
        "//machina/core/data:dataset_test_base",
        "//machina/core/data:dataset_utils",
        "//machina/core/kernels:function_ops",
        "//machina/core/kernels:identity_op",
    ],
)

tf_kernel_library(
    name = "iterator_ops",
    srcs = ["iterator_ops.cc"],
    hdrs = ["iterator_ops.h"],
    deps = [
        ":optional_ops",
        "//machina/core:core_cpu_internal",
        "//machina/core:dataset_ops_op_lib",
        "//machina/core:framework",
        "//machina/core:framework_internal",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core:protos_all_cc",
        "//machina/core:session_options",
        "//machina/core/activity_watcher",
        "//machina/core/activity_watcher:activity_watcher_utils",
        "//machina/core/data:captured_function",
        "//machina/core/data:dataset_utils",
        "//machina/core/data:finalization_utils",
        "//machina/core/data:metric_utils",
        "//machina/core/data:root_dataset",
        "//machina/core/data:serialization_utils",
        "//machina/core/data:tf_data_memory_logger",
        "//machina/core/data:tfdataz_metrics",
        "//machina/core/data:unbounded_thread_pool",
        "//machina/core/data:utils",
        "//machina/core/kernels:ops_util",
        "//machina/core/profiler/lib:traceme",
        "//machina/core/profiler/lib:traceme_encode",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/time",
    ],
)

tf_cc_test(
    name = "iterator_ops_test",
    size = "small",
    srcs = ["iterator_ops_test.cc"],
    deps = [
        ":iterator_ops",
        ":prefetch_dataset_op",
        ":range_dataset_op",
        "//machina/core:core_cpu_internal",
        "//machina/core:dataset_ops_op_lib",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core/data:dataset_test_base",
        "//machina/core/framework:function_testlib",
        "//machina/core/framework:types_proto_cc",
        "//machina/core/kernels:cwise_op",
        "//machina/core/kernels:function_ops",
        "//machina/core/lib/monitoring:cell_reader",
        "//machina/core/lib/monitoring:test_utils",
        "//machina/core/platform:errors",
        "//machina/core/platform:refcount",
        "//machina/core/platform:statusor",
    ],
)

tf_kernel_library(
    name = "map_dataset_op",
    srcs = ["map_dataset_op.cc"],
    hdrs = ["map_dataset_op.h"],
    deps = [
        "//machina/core:core_cpu_internal",
        "//machina/core:dataset_ops_op_lib",
        "//machina/core:framework",
        "//machina/core:lib_internal",
        "//machina/core/data:captured_function",
        "//machina/core/data:dataset_utils",
        "//machina/core/data:name_utils",
        "@com_google_absl//absl/status",
    ],
)

tf_cc_test(
    name = "map_dataset_op_test",
    size = "small",
    srcs = ["map_dataset_op_test.cc"],
    deps = [
        ":batch_dataset_op",
        ":iterator_ops",
        ":map_dataset_op",
        ":range_dataset_op",
        "//machina/core:core_cpu_internal",
        "//machina/core:dataset_ops_op_lib",
        "//machina/core:framework",
        "//machina/core:lib_internal",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core:testlib",
        "//machina/core/data:dataset_test_base",
        "//machina/core/data:dataset_utils",
        "//machina/core/data:stats_utils",
        "//machina/core/kernels:cwise_op",
        "//machina/core/kernels:function_ops",
    ],
)

tf_kernel_library(
    name = "map_defun_op",
    srcs = ["map_defun_op.cc"],
    hdrs = ["map_defun_op.h"],
    deps = [
        "//machina/core:core_cpu_internal",
        "//machina/core:framework",
        "//machina/core:functional_ops_op_lib",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core/data:dataset_utils",
    ],
)

tf_cc_test(
    name = "map_defun_op_test",
    size = "small",
    srcs = ["map_defun_op_test.cc"],
    deps = [
        ":map_defun_op",
        "//machina/core:core_cpu_internal",
        "//machina/core:dataset_ops_op_lib",
        "//machina/core:framework",
        "//machina/core:lib_internal",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core:testlib",
        "//machina/core/data:dataset_test_base",
        "//machina/core/data:dataset_utils",
        "//machina/core/data:stats_utils",
        "//machina/core/kernels:cwise_op",
        "//machina/core/kernels:function_ops",
    ],
)

tf_kernel_library(
    name = "model_dataset_op",
    srcs = ["model_dataset_op.cc"],
    hdrs = ["model_dataset_op.h"],
    deps = [
        "//machina/core:core_cpu_internal",
        "//machina/core:dataset_ops_op_lib",
        "//machina/core:framework",
        "//machina/core:framework_internal",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core/data:dataset_utils",
        "@com_google_absl//absl/memory",
    ],
)

tf_kernel_library(
    name = "multi_device_iterator_ops",
    srcs = ["multi_device_iterator_ops.cc"],
    deps = [
        ":iterator_ops",
        "//machina/core:core_cpu_internal",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core/data:dataset_utils",
        "//machina/core/data:finalization_utils",
        "//machina/core/data:metric_utils",
        "//machina/core/data:root_dataset",
        "//machina/core/data:tfdataz_metrics",
        "//machina/core/data:unbounded_thread_pool",
        "//machina/core/kernels:ops_util",
        "//machina/core/platform:env",
        "//machina/core/platform:refcount",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@local_tsl//tsl/profiler/lib:traceme",
        "@local_tsl//tsl/profiler/lib:traceme_encode",
    ],
)

tf_kernel_library(
    name = "optimize_dataset_op",
    srcs = ["optimize_dataset_op.cc"],
    hdrs = ["optimize_dataset_op.h"],
    deps = [
        "//machina/core:core_cpu_internal",
        "//machina/core:dataset_ops_op_lib",
        "//machina/core:framework",
        "//machina/core:lib_internal",
        "//machina/core:protos_all_cc",
        "//machina/core/data:dataset_utils",
        "//machina/core/data:rewrite_utils",
        "//machina/core/platform:platform_port",
        "@com_google_absl//absl/container:flat_hash_set",
    ],
)

tf_cc_test(
    name = "optimize_dataset_op_test",
    size = "small",
    srcs = ["optimize_dataset_op_test.cc"],
    deps = [
        ":iterator_ops",
        ":optimize_dataset_op",
        ":range_dataset_op",
        ":take_dataset_op",
        "//machina/core:core_cpu_internal",
        "//machina/core:dataset_ops_op_lib",
        "//machina/core:framework",
        "//machina/core:lib_internal",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core:testlib",
        "//machina/core/data:dataset_test_base",
        "//machina/core/data:dataset_utils",
        "//machina/core/kernels/data/experimental:threadpool_dataset_op",
    ],
)

tf_kernel_library(
    name = "optional_ops_util",
    srcs = ["optional_ops_util.cc"],
    hdrs = ["optional_ops_util.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//machina/core:framework",
        "//machina/core:lib",
    ],
)

tf_kernel_library(
    name = "optional_ops",
    srcs = ["optional_ops.cc"],
    hdrs = ["optional_ops.h"],
    gpu_srcs = [
        "optional_ops.cu.cc",
        "optional_ops.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":optional_ops_util",
        "//machina/core:core_cpu_internal",
        "//machina/core:dataset_ops_op_lib",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core:protos_all_cc",
        "@eigen_archive//:eigen3",
    ],
)

tf_kernel_library(
    name = "options_dataset_op",
    srcs = ["options_dataset_op.cc"],
    hdrs = ["options_dataset_op.h"],
    deps = [
        "//machina/core:dataset_ops_op_lib",
        "//machina/core:framework",
        "//machina/core/data:name_utils",
        "//machina/core/framework:dataset_options_proto_cc",
        "//machina/core/profiler/lib:traceme",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
    ],
)

tf_cc_test(
    name = "options_dataset_op_test",
    size = "small",
    srcs = ["options_dataset_op_test.cc"],
    deps = [
        ":options_dataset_op",
        ":range_dataset_op",
        "//machina/core:core_cpu_internal",
        "//machina/core:dataset_ops_op_lib",
        "//machina/core:framework",
        "//machina/core:lib_internal",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core:testlib",
        "//machina/core/data:dataset_test_base",
        "//machina/core/data:dataset_utils",
    ],
)

tf_kernel_library(
    name = "padded_batch_dataset_op",
    srcs = ["padded_batch_dataset_op.cc"],
    hdrs = ["padded_batch_dataset_op.h"],
    deps = [
        "//machina/core:dataset_ops_op_lib",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core/data:dataset_utils",
        "//machina/core/data:name_utils",
    ],
)

tf_cc_test(
    name = "padded_batch_dataset_op_test",
    size = "small",
    srcs = ["padded_batch_dataset_op_test.cc"],
    deps = [
        ":concatenate_dataset_op",
        ":iterator_ops",
        ":padded_batch_dataset_op",
        ":range_dataset_op",
        ":tensor_slice_dataset_op",
        "//machina/core:core_cpu_internal",
        "//machina/core:dataset_ops_op_lib",
        "//machina/core:framework",
        "//machina/core:lib_internal",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core:testlib",
        "//machina/core/data:dataset_test_base",
        "//machina/core/data:dataset_utils",
    ],
)

tf_kernel_library(
    name = "parallel_batch_dataset_op",
    srcs = ["parallel_batch_dataset_op.cc"],
    hdrs = ["parallel_batch_dataset_op.h"],
    deps = [
        "//machina/core:core_cpu_internal",
        "//machina/core:dataset_ops_op_lib",
        "//machina/core:framework",
        "//machina/core:framework_internal",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core:protos_all_cc",
        "//machina/core/data:dataset_utils",
        "//machina/core/data:name_utils",
        "//machina/core/data:stats_utils",
        "//machina/core/profiler/lib:traceme",
        "//machina/core/profiler/lib:traceme_encode",
        "@com_google_absl//absl/log",
    ],
)

tf_cc_test(
    name = "parallel_batch_dataset_op_test",
    size = "small",
    srcs = ["parallel_batch_dataset_op_test.cc"],
    deps = [
        ":iterator_ops",
        ":parallel_batch_dataset_op",
        ":range_dataset_op",
        "//machina/core:core_cpu_internal",
        "//machina/core:dataset_ops_op_lib",
        "//machina/core:framework",
        "//machina/core:lib_internal",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core:testlib",
        "//machina/core/data:dataset_test_base",
        "//machina/core/data:dataset_utils",
        "//machina/core/data:name_utils",
        "//machina/core/data:stats_utils",
        "//machina/core/kernels:cwise_op",
        "//machina/core/kernels:function_ops",
    ],
)

tf_kernel_library(
    name = "parallel_interleave_dataset_op",
    srcs = ["parallel_interleave_dataset_op.cc"],
    hdrs = ["parallel_interleave_dataset_op.h"],
    deps = [
        "//machina/core:core_cpu_internal",
        "//machina/core:dataset_ops_op_lib",
        "//machina/core:framework",
        "//machina/core:framework_internal",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core/data:captured_function",
        "//machina/core/data:dataset_utils",
        "//machina/core/data:name_utils",
        "//machina/core/data:stats_utils",
        "//machina/core/profiler/lib:traceme",
        "//machina/core/profiler/lib:traceme_encode",
        "@com_google_absl//absl/strings:str_format",
    ],
)

tf_cc_test(
    name = "parallel_interleave_dataset_op_test",
    size = "small",
    srcs = ["parallel_interleave_dataset_op_test.cc"],
    deps = [
        ":iterator_ops",
        ":parallel_interleave_dataset_op",
        ":tensor_slice_dataset_op",
        "//machina/core:core_cpu_internal",
        "//machina/core:dataset_ops_op_lib",
        "//machina/core:framework",
        "//machina/core:graph",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core:testlib",
        "//machina/core/data:captured_function",
        "//machina/core/data:dataset_test_base",
        "//machina/core/data:dataset_utils",
        "//machina/core/kernels:function_ops",
        "//machina/core/kernels:identity_op",
    ],
)

tf_kernel_library(
    name = "parallel_map_dataset_op",
    srcs = ["parallel_map_dataset_op.cc"],
    hdrs = ["parallel_map_dataset_op.h"],
    deps = [
        "//machina/core:core_cpu_internal",
        "//machina/core:dataset_ops_op_lib",
        "//machina/core:framework",
        "//machina/core:framework_internal",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core:protos_all_cc",
        "//machina/core/data:captured_function",
        "//machina/core/data:dataset_utils",
        "//machina/core/data:name_utils",
        "//machina/core/data:stats_utils",
        "//machina/core/data:unbounded_thread_pool",
        "//machina/core/profiler/lib:traceme",
        "//machina/core/profiler/lib:traceme_encode",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:str_format",
        "@local_xla//xla/tsl/platform:logging",
    ],
)

tf_cc_test(
    name = "parallel_map_dataset_op_test",
    size = "small",
    srcs = ["parallel_map_dataset_op_test.cc"],
    deps = [
        ":batch_dataset_op",
        ":iterator_ops",
        ":parallel_map_dataset_op",
        ":range_dataset_op",
        "//machina/core:core_cpu_internal",
        "//machina/core:dataset_ops_op_lib",
        "//machina/core:framework",
        "//machina/core:lib_internal",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core:testlib",
        "//machina/core/data:dataset_test_base",
        "//machina/core/data:dataset_utils",
        "//machina/core/data:name_utils",
        "//machina/core/data:stats_utils",
        "//machina/core/kernels:cwise_op",
        "//machina/core/kernels:function_ops",
        "@com_google_googletest//:gtest",
    ],
)

tf_kernel_library(
    name = "parallel_filter_dataset_op",
    srcs = ["parallel_filter_dataset_op.cc"],
    hdrs = ["parallel_filter_dataset_op.h"],
    deps = [
        "//machina/core:core_cpu_internal",
        "//machina/core:dataset_ops_op_lib",
        "//machina/core:framework",
        "//machina/core:framework_internal",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core:protos_all_cc",
        "//machina/core/data:captured_function",
        "//machina/core/data:dataset_utils",
        "//machina/core/data:name_utils",
        "//machina/core/data:stats_utils",
        "//machina/core/profiler/lib:traceme",
        "//machina/core/profiler/lib:traceme_encode",
        "@com_google_absl//absl/status",
    ],
)

tf_cc_test(
    name = "parallel_filter_dataset_op_test",
    size = "small",
    srcs = ["parallel_filter_dataset_op_test.cc"],
    deps = [
        ":iterator_ops",
        ":parallel_filter_dataset_op",
        ":tensor_slice_dataset_op",
        "//machina/core:core_cpu_internal",
        "//machina/core:dataset_ops_op_lib",
        "//machina/core:framework",
        "//machina/core:lib_internal",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core:testlib",
        "//machina/core/data:dataset_test_base",
        "//machina/core/data:dataset_utils",
        "//machina/core/data:name_utils",
        "//machina/core/data:stats_utils",
        "//machina/core/kernels:cwise_op",
        "//machina/core/kernels:function_ops",
        "//machina/core/kernels:unique_op",
    ],
)

cc_library(
    name = "prefetch_autotuner",
    srcs = ["prefetch_autotuner.cc"],
    hdrs = ["prefetch_autotuner.h"],
    deps = [
        "//machina/core:framework_internal",
        "//machina/core:lib",
        "//machina/core/data:dataset_utils",
        "@com_google_absl//absl/types:optional",
    ],
)

tf_cc_test(
    name = "prefetch_autotuner_test",
    size = "small",
    srcs = ["prefetch_autotuner_test.cc"],
    deps = [
        ":prefetch_autotuner",
        "//machina/core:framework_internal",
        "//machina/core:test",
        "//machina/core:test_main",
    ],
)

tf_kernel_library(
    name = "prefetch_dataset_op",
    srcs = ["prefetch_dataset_op.cc"],
    hdrs = ["prefetch_dataset_op.h"],
    deps = [
        ":prefetch_autotuner",
        "//machina/core:core_cpu_internal",
        "//machina/core:dataset_ops_op_lib",
        "//machina/core:framework",
        "//machina/core:framework_internal",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core:protos_all_cc",
        "//machina/core/data:dataset_utils",
        "//machina/core/data:name_utils",
        "//machina/core/data:stats_utils",
        "//machina/core/profiler/lib:traceme",
        "//machina/core/profiler/lib:traceme_encode",
        "@com_google_absl//absl/status",
        "@local_tsl//tsl/platform:mutex",
    ],
)

tf_cc_test(
    name = "prefetch_dataset_op_test",
    size = "small",
    srcs = ["prefetch_dataset_op_test.cc"],
    deps = [
        ":iterator_ops",
        ":prefetch_dataset_op",
        ":tensor_slice_dataset_op",
        "//machina/core:core_cpu_internal",
        "//machina/core:dataset_ops_op_lib",
        "//machina/core:framework",
        "//machina/core:lib_internal",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core:testlib",
        "//machina/core/data:dataset_test_base",
        "//machina/core/data:dataset_utils",
    ],
)

tf_kernel_library(
    name = "random_seed_ops",
    srcs = ["random_seed_ops.cc"],
    hdrs = ["random_seed_ops.h"],
    deps = [
        "//machina/core:core_cpu_internal",
        "//machina/core:framework",
        "//machina/core:functional_ops_op_lib",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core/data:dataset_utils",
    ],
)

tf_kernel_library(
    name = "range_dataset_op",
    srcs = ["range_dataset_op.cc"],
    hdrs = ["range_dataset_op.h"],
    deps = [
        "//machina/core:dataset_ops_op_lib",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core/data:global_shuffle_utils",
        "//machina/core/data:name_utils",
        "//machina/core/data:split_utils",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@local_tsl//tsl/platform:mutex",
        "@local_xla//xla/tsl/platform:types",
    ],
)

tf_cc_test(
    name = "range_dataset_op_test",
    size = "small",
    srcs = ["range_dataset_op_test.cc"],
    deps = [
        ":iterator_ops",
        ":range_dataset_op",
        "//machina/core:framework",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core:testlib",
        "//machina/core/data:dataset_test_base",
        "//machina/core/data:dataset_utils",
    ],
)

tf_kernel_library(
    name = "reduce_dataset_op",
    srcs = ["reduce_dataset_op.cc"],
    hdrs = ["reduce_dataset_op.h"],
    deps = [
        ":iterator_ops",
        ":prefetch_dataset_op",
        "//machina/core:core_cpu_internal",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core/data:captured_function",
        "//machina/core/data:root_dataset",
        "//machina/core/profiler/lib:traceme",
    ],
)

tf_cc_test(
    name = "reduce_dataset_op_test",
    size = "small",
    srcs = ["reduce_dataset_op_test.cc"],
    deps = [
        ":range_dataset_op",
        ":reduce_dataset_op",
        "//machina/core:core_cpu_internal",
        "//machina/core:dataset_ops_op_lib",
        "//machina/core:framework",
        "//machina/core:lib_internal",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core:testlib",
        "//machina/core/data:dataset_test_base",
        "//machina/core/data:dataset_utils",
        "//machina/core/data:stats_utils",
        "//machina/core/kernels:cwise_op",
        "//machina/core/kernels:function_ops",
    ],
)

tf_kernel_library(
    name = "repeat_dataset_op",
    srcs = ["repeat_dataset_op.cc"],
    hdrs = ["repeat_dataset_op.h"],
    deps = [
        "//machina/core:dataset_ops_op_lib",
        "//machina/core:framework",
        "//machina/core:lib_internal",
        "//machina/core/data:global_shuffle_utils",
        "//machina/core/data:name_utils",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:thread_annotations",
        "@local_xla//xla/tsl/platform:errors",
    ],
)

tf_cc_test(
    name = "repeat_dataset_op_test",
    size = "small",
    srcs = ["repeat_dataset_op_test.cc"],
    deps = [
        ":iterator_ops",
        ":repeat_dataset_op",
        ":tensor_slice_dataset_op",
        "//machina/core:core_cpu_internal",
        "//machina/core:dataset_ops_op_lib",
        "//machina/core:framework",
        "//machina/core:lib_internal",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core:testlib",
        "//machina/core/data:dataset_test_base",
        "//machina/core/data:dataset_utils",
        "//machina/core/data:serialization_utils",
    ],
)

tf_kernel_library(
    name = "rewrite_dataset_op",
    srcs = ["rewrite_dataset_op.cc"],
    hdrs = ["rewrite_dataset_op.h"],
    deps = [
        "//machina/core:core_cpu_internal",
        "//machina/core:dataset_ops_op_lib",
        "//machina/core:framework",
        "//machina/core:lib_internal",
        "//machina/core:protos_all_cc",
        "//machina/core/data:dataset_utils",
        "//machina/core/data:rewrite_utils",
        "//machina/core/platform:platform_port",
        "@com_google_absl//absl/container:flat_hash_set",
    ],
)

tf_cc_test(
    name = "rewrite_dataset_op_test",
    size = "small",
    srcs = ["rewrite_dataset_op_test.cc"],
    deps = [
        ":iterator_ops",
        ":range_dataset_op",
        ":rewrite_dataset_op",
        "//machina/core:core_cpu_internal",
        "//machina/core:dataset_ops_op_lib",
        "//machina/core:framework",
        "//machina/core:lib_internal",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core:testlib",
        "//machina/core/data:dataset_test_base",
        "//machina/core/data:dataset_utils",
    ],
)

tf_kernel_library(
    name = "shard_dataset_op",
    srcs = ["shard_dataset_op.cc"],
    hdrs = ["shard_dataset_op.h"],
    deps = [
        "//machina/core:dataset_ops_op_lib",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core/data:dataset_utils",
        "//machina/core/data:global_shuffle_utils",
        "//machina/core/data:name_utils",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:thread_annotations",
    ],
)

tf_cc_test(
    name = "shard_dataset_op_test",
    size = "small",
    srcs = ["shard_dataset_op_test.cc"],
    deps = [
        ":iterator_ops",
        ":range_dataset_op",
        ":shard_dataset_op",
        "//machina/core:core_cpu_internal",
        "//machina/core:dataset_ops_op_lib",
        "//machina/core:framework",
        "//machina/core:lib_internal",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core:testlib",
        "//machina/core/data:dataset_test_base",
        "//machina/core/data:dataset_utils",
    ],
)

tf_kernel_library(
    name = "shuffle_dataset_op",
    srcs = ["shuffle_dataset_op.cc"],
    hdrs = ["shuffle_dataset_op.h"],
    deps = [
        ":random_seed_ops",
        "//machina/core:dataset_ops_op_lib",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core/data:dataset_utils",
        "//machina/core/data:name_utils",
        "//machina/core/data:serialization_utils",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "shuffle_dataset_op_test",
    size = "small",
    srcs = ["shuffle_dataset_op_test.cc"],
    deps = [
        "shuffle_dataset_op",
        ":iterator_ops",
        ":range_dataset_op",
        "//machina/core:framework",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core:testlib",
        "//machina/core/data:dataset_test_base",
        "//machina/core/data:dataset_utils",
        "//machina/core/data:serialization_utils",
    ],
)

tf_kernel_library(
    name = "skip_dataset_op",
    srcs = ["skip_dataset_op.cc"],
    hdrs = ["skip_dataset_op.h"],
    deps = [
        "//machina/core:dataset_ops_op_lib",
        "//machina/core:framework",
        "//machina/core/data:global_shuffle_utils",
        "//machina/core/data:name_utils",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@local_xla//xla/tsl/platform:errors",
        "@local_xla//xla/tsl/platform:statusor",
    ],
)

tf_cc_test(
    name = "skip_dataset_op_test",
    size = "small",
    srcs = ["skip_dataset_op_test.cc"],
    deps = [
        ":iterator_ops",
        ":range_dataset_op",
        ":skip_dataset_op",
        ":tensor_slice_dataset_op",
        "//machina/core:core_cpu_internal",
        "//machina/core:dataset_ops_op_lib",
        "//machina/core:framework",
        "//machina/core:lib_internal",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core:testlib",
        "//machina/core/data:dataset_test_base",
        "//machina/core/data:dataset_utils",
    ],
)

tf_kernel_library(
    name = "sparse_tensor_slice_dataset_op",
    srcs = ["sparse_tensor_slice_dataset_op.cc"],
    deps = [
        "//machina/core:dataset_ops_op_lib",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:lib_internal",
    ],
)

tf_cc_test(
    name = "sparse_tensor_slice_dataset_op_test",
    size = "small",
    srcs = ["sparse_tensor_slice_dataset_op_test.cc"],
    deps = [
        ":iterator_ops",
        ":sparse_tensor_slice_dataset_op",
        "//machina/core:core_cpu_internal",
        "//machina/core:dataset_ops_op_lib",
        "//machina/core:framework",
        "//machina/core:lib_internal",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core:testlib",
        "//machina/core/data:dataset_test_base",
        "//machina/core/data:dataset_utils",
        "//machina/core/data:serialization_utils",
    ],
)

tf_kernel_library(
    name = "take_dataset_op",
    srcs = ["take_dataset_op.cc"],
    hdrs = ["take_dataset_op.h"],
    deps = [
        "//machina/core:dataset_ops_op_lib",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core/data:name_utils",
        "@com_google_absl//absl/status",
    ],
)

tf_cc_test(
    name = "take_dataset_op_test",
    size = "small",
    srcs = ["take_dataset_op_test.cc"],
    deps = [
        ":iterator_ops",
        ":range_dataset_op",
        ":take_dataset_op",
        ":tensor_slice_dataset_op",
        "//machina/core:core_cpu_internal",
        "//machina/core:dataset_ops_op_lib",
        "//machina/core:framework",
        "//machina/core:lib_internal",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core:testlib",
        "//machina/core/data:dataset_test_base",
        "//machina/core/data:dataset_utils",
    ],
)

tf_kernel_library(
    name = "tensor_dataset_op",
    srcs = ["tensor_dataset_op.cc"],
    hdrs = ["tensor_dataset_op.h"],
    deps = [
        "//machina/core:dataset_ops_op_lib",
        "//machina/core:framework",
        "//machina/core:graph",
        "//machina/core/data:dataset_utils",
        "//machina/core/data:global_shuffle_utils",
        "//machina/core/data:name_utils",
        "//machina/core/data:split_utils",
        "@com_google_absl//absl/status",
        "@local_tsl//tsl/platform:mutex",
        "@local_tsl//tsl/platform:thread_annotations",
    ],
)

tf_cc_test(
    name = "tensor_dataset_op_test",
    size = "small",
    srcs = ["tensor_dataset_op_test.cc"],
    deps = [
        ":tensor_dataset_op",
        "//machina/core:test_main",
        "//machina/core/data:dataset_test_base",
        "//machina/core/data:dataset_utils",
        "//machina/core/data:serialization_utils",
    ],
)

tf_kernel_library(
    name = "tensor_slice_dataset_op",
    srcs = ["tensor_slice_dataset_op.cc"],
    hdrs = ["tensor_slice_dataset_op.h"],
    deps = [
        "//machina/core:dataset_ops_op_lib",
        "//machina/core:framework",
        "//machina/core:graph",
        "//machina/core/data:dataset_utils",
        "//machina/core/data:global_shuffle_utils",
        "//machina/core/data:name_utils",
        "//machina/core/data:split_utils",
        "@com_google_absl//absl/status",
        "@local_tsl//tsl/platform:mutex",
        "@local_tsl//tsl/platform:thread_annotations",
    ],
)

tf_cc_test(
    name = "tensor_slice_dataset_op_test",
    size = "small",
    srcs = ["tensor_slice_dataset_op_test.cc"],
    deps = [
        ":iterator_ops",
        ":tensor_slice_dataset_op",
        "//machina/core:core_cpu_internal",
        "//machina/core:dataset_ops_op_lib",
        "//machina/core:framework",
        "//machina/core:lib_internal",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core:testlib",
        "//machina/core/data:dataset_test_base",
        "//machina/core/data:dataset_utils",
        "//machina/core/data:serialization_utils",
    ],
)

tf_kernel_library(
    name = "text_line_dataset_op",
    srcs = ["text_line_dataset_op.cc"],
    hdrs = ["text_line_dataset_op.h"],
    deps = [
        "//machina/core:core_cpu_internal",
        "//machina/core:dataset_ops_op_lib",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core/data:name_utils",
        "//machina/core/data:utils",
    ],
)

tf_cc_test(
    name = "text_line_dataset_op_test",
    size = "small",
    srcs = ["text_line_dataset_op_test.cc"],
    deps = [
        ":iterator_ops",
        ":text_line_dataset_op",
        "//machina/core:core_cpu_internal",
        "//machina/core:dataset_ops_op_lib",
        "//machina/core:test_main",
        "//machina/core/data:dataset_test_base",
    ],
)

tf_kernel_library(
    name = "tf_record_dataset_op",
    srcs = ["tf_record_dataset_op.cc"],
    hdrs = ["tf_record_dataset_op.h"],
    deps = [
        "//machina/core:core_cpu_internal",
        "//machina/core:dataset_ops_op_lib",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core/data:name_utils",
        "//machina/core/data:utils",
        "@local_tsl//tsl/profiler/lib:traceme",
    ],
)

tf_cc_test(
    name = "tf_record_dataset_op_test",
    size = "small",
    srcs = ["tf_record_dataset_op_test.cc"],
    deps = [
        ":iterator_ops",
        ":tf_record_dataset_op",
        "//machina/core:core_cpu_internal",
        "//machina/core:dataset_ops_op_lib",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core/data:dataset_test_base",
        "//machina/core/data:name_utils",
        "//machina/core/framework:types_proto_cc",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
        "@local_xla//xla/tsl/lib/core:status_test_util",
    ],
)

cc_library(
    name = "window_dataset",
    srcs = ["window_dataset.cc"],
    hdrs = ["window_dataset.h"],
    deps = [
        "//machina/core:framework",
        "//machina/core:graph",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core/data:name_utils",
        "//machina/core/framework:dataset_options_proto_cc",
        "@com_google_absl//absl/status",
    ],
)

tf_kernel_library(
    name = "window_dataset_op",
    srcs = ["window_dataset_op.cc"],
    hdrs = ["window_dataset_op.h"],
    deps = [
        ":window_dataset",
        "//machina/core:dataset_ops_op_lib",
        "//machina/core:framework",
        "//machina/core:framework_internal",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core/data:name_utils",
        "//machina/core/framework:dataset_options_proto_cc",
        "//machina/core/framework:types_proto_cc",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
    ],
)

tf_cc_test(
    name = "window_dataset_op_test",
    size = "small",
    srcs = ["window_dataset_op_test.cc"],
    deps = [
        ":iterator_ops",
        ":range_dataset_op",
        ":window_dataset_op",
        "//machina/core:core_cpu_internal",
        "//machina/core:dataset_ops_op_lib",
        "//machina/core:framework",
        "//machina/core:lib_internal",
        "//machina/core:portable_gif_internal",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core/data:dataset_test_base",
        "//machina/core/data:name_utils",
        "//machina/core/data:serialization_utils",
        "//machina/core/framework:types_proto_cc",
        "//machina/core/platform:status",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
    ],
)

tf_kernel_library(
    name = "zip_dataset_op",
    srcs = ["zip_dataset_op.cc"],
    hdrs = ["zip_dataset_op.h"],
    deps = [
        "//machina/core:dataset_ops_op_lib",
        "//machina/core:framework",
        "//machina/core:framework_internal",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core/data:name_utils",
        "//machina/core/data:split_utils",
        "//machina/core/framework:dataset_options_proto_cc",
        "//machina/core/framework:types_proto_cc",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
    ],
)

tf_cc_test(
    name = "zip_dataset_op_test",
    size = "small",
    srcs = ["zip_dataset_op_test.cc"],
    deps = [
        ":iterator_ops",
        ":range_dataset_op",
        ":zip_dataset_op",
        "//machina/core:core_cpu_internal",
        "//machina/core:dataset_ops_op_lib",
        "//machina/core:framework",
        "//machina/core:lib_internal",
        "//machina/core:portable_gif_internal",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core/data:dataset_test_base",
        "//machina/core/data:name_utils",
        "//machina/core/framework:types_proto_cc",
        "//machina/core/platform:status",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ],
)

filegroup(
    name = "portable_all_op_kernels_headers",
    srcs = [
        "//machina/core/data:captured_function.h",
        "//machina/core/data:compression_utils.h",
        "//machina/core/data:dataset_utils.h",
        "//machina/core/data:finalization_utils.h",
        "//machina/core/data:flat_map_utils.h",
        "//machina/core/data:global_shuffle_utils.h",
        "//machina/core/data:metric_utils.h",
        "//machina/core/data:name_utils.h",
        "//machina/core/data:rewrite_utils.h",
        "//machina/core/data:root_dataset.h",
        "//machina/core/data:serialization_utils.h",
        "//machina/core/data:split_utils.h",
        "//machina/core/data:stats_utils.h",
        "//machina/core/data:tf_data_memory_logger.h",
        "//machina/core/data:tfdataz_metrics.h",
        "//machina/core/data:unbounded_thread_pool.h",
        "//machina/core/data:utils.h",
        "//machina/core/kernels/data/experimental:portable_all_op_kernels_headers",
    ] + glob(
        [
            "*.h",
        ],
    ),
    visibility = ["//machina:__subpackages__"],
)

# A file group which contains all operators which are known to work on mobile.
filegroup(
    name = "portable_all_op_kernels",
    srcs = [
        ":portable_all_op_kernels_headers",
        "//machina/core/data:captured_function.cc",
        "//machina/core/data:compression_utils.cc",
        "//machina/core/data:dataset_utils.cc",
        "//machina/core/data:finalization_utils.cc",
        "//machina/core/data:flat_map_utils.cc",
        "//machina/core/data:global_shuffle_utils.cc",
        "//machina/core/data:metric_utils.cc",
        "//machina/core/data:name_utils.cc",
        "//machina/core/data:rewrite_utils.cc",
        "//machina/core/data:root_dataset.cc",
        "//machina/core/data:serialization_utils.cc",
        "//machina/core/data:split_utils.cc",
        "//machina/core/data:stats_utils.cc",
        "//machina/core/data:tf_data_memory_logger.cc",
        "//machina/core/data:tfdataz_metrics.cc",
        "//machina/core/data:unbounded_thread_pool.cc",
        "//machina/core/data:utils.cc",
        "//machina/core/kernels/data/experimental:portable_all_op_kernels",
    ] + glob(
        [
            "*.cc",
        ],
        exclude = [
            "*test.cc",
        ],
    ),
    visibility = ["//machina:__subpackages__"],
)

tf_kernel_library(
    name = "data",
    visibility = ["//visibility:public"],
    deps = [
        ":batch_dataset_op",
        ":cache_dataset_ops",
        ":concatenate_dataset_op",
        ":dataset_ops",
        ":filter_dataset_op",
        ":finalize_dataset_op",
        ":fixed_length_record_dataset_op",
        ":flat_map_dataset_op",
        ":generator_dataset_op",
        ":get_options_op",
        ":interleave_dataset_op",
        ":iterator_ops",
        ":map_dataset_op",
        ":map_defun_op",
        ":model_dataset_op",
        ":multi_device_iterator_ops",
        ":optimize_dataset_op",
        ":optional_ops",
        ":options_dataset_op",
        ":padded_batch_dataset_op",
        ":parallel_batch_dataset_op",
        ":parallel_filter_dataset_op",
        ":parallel_interleave_dataset_op",
        ":parallel_map_dataset_op",
        ":prefetch_dataset_op",
        ":range_dataset_op",
        ":reduce_dataset_op",
        ":repeat_dataset_op",
        ":rewrite_dataset_op",
        ":shard_dataset_op",
        ":shuffle_dataset_op",
        ":skip_dataset_op",
        ":sparse_tensor_slice_dataset_op",
        ":take_dataset_op",
        ":tensor_dataset_op",
        ":tensor_slice_dataset_op",
        ":text_line_dataset_op",
        ":tf_record_dataset_op",
        ":window_dataset_op",
        ":zip_dataset_op",
        "//machina/core:array_ops_op_lib",
        "//machina/core:nn_ops_op_lib",
        "//machina/core/kernels/data/experimental",
    ],
)

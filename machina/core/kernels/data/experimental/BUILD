# Description:
#   Contains experimental kernels for datasets and iterators.

load(
    "//machina:machina.bzl",
    "if_not_mobile",
    "tf_cc_test",
)
load("//machina:machina.default.bzl", "filegroup", "tf_kernel_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//machina:internal"],
    licenses = ["notice"],
)

filegroup(
    name = "portable_all_op_kernels_headers",
    srcs = [
        "threadpool_dataset_op.h",
    ],
    visibility = ["//machina:__subpackages__"],
)

# A file group which contains all operators which are known to work on mobile.
filegroup(
    name = "portable_all_op_kernels",
    srcs = [
        "lookup_ops.cc",
        "take_while_dataset_op.cc",
        "threadpool_dataset_op.cc",
        "unbatch_dataset_op.cc",
        ":portable_all_op_kernels_headers",
    ],
    visibility = ["//machina:__subpackages__"],
)

tf_kernel_library(
    name = "assert_cardinality_dataset_op",
    srcs = ["assert_cardinality_dataset_op.cc"],
    hdrs = ["assert_cardinality_dataset_op.h"],
    deps = [
        "//machina/core:experimental_dataset_ops_op_lib",
        "//machina/core:framework",
        "//machina/core/data:name_utils",
        "//machina/core/platform:errors",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:errors",
    ],
)

tf_kernel_library(
    name = "assert_next_dataset_op",
    srcs = ["assert_next_dataset_op.cc"],
    hdrs = ["assert_next_dataset_op.h"],
    deps = [
        "//machina/core:experimental_dataset_ops_op_lib",
        "//machina/core:framework",
        "//machina/core/data:dataset_utils",
        "//machina/core/data:name_utils",
    ],
)

tf_cc_test(
    name = "assert_next_dataset_op_test",
    size = "small",
    srcs = ["assert_next_dataset_op_test.cc"],
    deps = [
        ":assert_next_dataset_op",
        "//machina/core:experimental_dataset_ops_op_lib",
        "//machina/core:test_main",
        "//machina/core/data:dataset_test_base",
        "//machina/core/kernels/data:range_dataset_op",
        "//machina/core/kernels/data:take_dataset_op",
    ],
)

tf_kernel_library(
    name = "assert_prev_dataset_op",
    srcs = ["assert_prev_dataset_op.cc"],
    hdrs = ["assert_prev_dataset_op.h"],
    deps = [
        "//machina/core:experimental_dataset_ops_op_lib",
        "//machina/core:framework",
        "//machina/core:graph",
        "//machina/core:protos_all_cc",
        "//machina/core/data:dataset_utils",
        "//machina/core/data:name_utils",
        "//machina/core/data:rewrite_utils",
        "//machina/core/data:serialization_utils",
        "//machina/core/platform:errors",
        "//machina/core/platform:protobuf",
    ],
)

tf_cc_test(
    name = "assert_prev_dataset_op_test",
    size = "small",
    srcs = ["assert_prev_dataset_op_test.cc"],
    deps = [
        ":assert_prev_dataset_op",
        "//machina/core:experimental_dataset_ops_op_lib",
        "//machina/core:test_main",
        "//machina/core/data:dataset_test_base",
        "//machina/core/kernels/data:range_dataset_op",
        "//machina/core/kernels/data:take_dataset_op",
        "//machina/core/kernels/data:tensor_slice_dataset_op",
        "@com_google_absl//absl/strings",
    ],
)

tf_kernel_library(
    name = "auto_shard_dataset_op",
    srcs = ["auto_shard_dataset_op.cc"],
    hdrs = ["auto_shard_dataset_op.h"],
    deps = [
        "//machina/core:core_cpu_internal",
        "//machina/core:dataset_ops_op_lib",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core:protos_all_cc",
        "//machina/core/data:rewrite_utils",
        "//machina/core/grappler/optimizers/data:auto_shard",
    ],
)

tf_cc_test(
    name = "auto_shard_dataset_op_test",
    size = "small",
    srcs = ["auto_shard_dataset_op_test.cc"],
    deps = [
        ":auto_shard_dataset_op",
        "//machina/core:experimental_dataset_ops_op_lib",
        "//machina/core:framework",
        "//machina/core:graph",
        "//machina/core:session_options",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core:testlib",
        "//machina/core/common_runtime:type_inference",
        "//machina/core/data:dataset_test_base",
        "//machina/core/kernels/data:range_dataset_op",
        "//machina/core/kernels/data:shard_dataset_op",
        "@eigen_archive//:eigen3",
    ],
)

tf_kernel_library(
    name = "choose_fastest_branch_dataset_op",
    srcs = ["choose_fastest_branch_dataset_op.cc"],
    deps = [
        "//machina/core:experimental_dataset_ops_op_lib",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core:protos_all_cc",
        "//machina/core/data:captured_function",
        "//machina/core/data:dataset_utils",
        "//machina/core/kernels/data:take_dataset_op",
    ],
)

tf_kernel_library(
    name = "check_pinned_op",
    srcs = ["check_pinned_op.cc"],
    deps = [
        "//machina/core:framework",
        "@com_google_absl//absl/status",
    ],
)

tf_kernel_library(
    name = "choose_fastest_dataset_op",
    srcs = ["choose_fastest_dataset_op.cc"],
    deps = [
        "//machina/core:experimental_dataset_ops_op_lib",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:lib_internal",
    ],
)

tf_kernel_library(
    name = "compression_ops",
    srcs = ["compression_ops.cc"],
    hdrs = ["compression_ops.h"],
    deps = [
        "//machina/core:experimental_dataset_ops_op_lib",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core/data:compression_utils",
    ],
)

tf_kernel_library(
    name = "compute_batch_size_op",
    srcs = ["compute_batch_size_op.cc"],
    deps = [
        "//machina/core:core_cpu_internal",
        "//machina/core:dataset_ops_op_lib",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core:protos_all_cc",
        "//machina/core/data:dataset_utils",
        "//machina/core/data:name_utils",
        "//machina/core/data:serialization_utils",
        "//machina/core/grappler:graph_view",
        "//machina/core/grappler/optimizers/data:graph_utils",
        "//machina/core/platform:regexp",
    ],
)

tf_kernel_library(
    name = "csv_dataset_op",
    srcs = ["csv_dataset_op.cc"],
    deps = [
        "//machina/core:experimental_dataset_ops_op_lib",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:lib_internal",
    ],
)

tf_kernel_library(
    name = "data_service_dataset_op",
    srcs = ["data_service_dataset_op.cc"],
    hdrs = ["data_service_dataset_op.h"],
    deps = [
        "//machina/core:experimental_dataset_ops_op_lib",
        "//machina/core:framework",
        "//machina/core:framework_internal",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core:protos_all_cc",
        "//machina/core/data:captured_function",
        "//machina/core/data:dataset_utils",
        "//machina/core/data:name_utils",
        "//machina/core/data:utils",
        "//machina/core/data/service:common",
        "//machina/core/data/service:common_proto_cc",
        "//machina/core/data/service:dispatcher_proto_cc",
        "//machina/core/data/service/client:common",
        "//machina/core/data/service/client:data_service_client",
        "//machina/core/data/service/client:utils",
        "//machina/core/kernels/data:parallel_map_dataset_op",
        "//machina/core/platform:errors",
        "//machina/core/platform:refcount",
        "//machina/core/platform:status",
        "//machina/core/platform:statusor",
        "//machina/core/platform:tstring",
        "//machina/core/platform:types",
        "//machina/core/profiler/lib:traceme",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
    ],
)

tf_kernel_library(
    name = "distributed_save_op",
    srcs = ["distributed_save_op.cc"],
    hdrs = ["distributed_save_op.h"],
    deps = [
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core/data:serialization_utils",
        "//machina/core/data/service:common_proto_cc",
        "//machina/core/data/service:dispatcher_client",
        "//machina/core/data/service:grpc_util",
        "//machina/core/data/service:py_utils",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
    ],
)

tf_kernel_library(
    name = "data_service_ops",
    srcs = ["data_service_ops.cc"],
    hdrs = ["data_service_ops.h"],
    deps = [
        "//machina/core:experimental_dataset_ops_op_lib",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core:protos_all_cc",
        "//machina/core/data:serialization_utils",
        "//machina/core/data/service:common_proto_cc",
        "//machina/core/data/service:dispatcher_client",
        "//machina/core/data/service:grpc_util",
        "//machina/core/framework:graph_proto_cc",
        "//machina/core/platform:env_time",
        "//machina/core/platform:errors",
        "//machina/core/platform:status",
        "//machina/core/platform:tstring",
        "@com_google_absl//absl/strings",
    ],
)

tf_kernel_library(
    name = "dense_to_sparse_batch_dataset_op",
    srcs = ["dense_to_sparse_batch_dataset_op.cc"],
    deps = [
        "//machina/core:experimental_dataset_ops_op_lib",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:lib_internal",
    ],
)

tf_kernel_library(
    name = "directed_interleave_dataset_op",
    srcs = ["directed_interleave_dataset_op.cc"],
    hdrs = ["directed_interleave_dataset_op.h"],
    deps = [
        "//machina/core:experimental_dataset_ops_op_lib",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core/data:name_utils",
        "//machina/core/data:split_utils",
        "@eigen_archive//:eigen3",
    ],
)

tf_cc_test(
    name = "directed_interleave_dataset_op_test",
    size = "small",
    srcs = ["directed_interleave_dataset_op_test.cc"],
    deps = [
        ":directed_interleave_dataset_op",
        "//machina/core:experimental_dataset_ops_op_lib",
        "//machina/core:test_main",
        "//machina/core:testlib",
        "//machina/core/data:dataset_test_base",
        "//machina/core/kernels/data:range_dataset_op",
        "//machina/core/kernels/data:tensor_slice_dataset_op",
    ],
)

tf_kernel_library(
    name = "global_shuffle_dataset_op",
    srcs = ["global_shuffle_dataset_op.cc"],
    deps = [
        "//machina/core:dataset_ops_op_lib",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core/data:name_utils",
        "//machina/core/framework:dataset_options_proto_cc",
        "//machina/core/framework:types_proto_cc",
        "//machina/core/kernels:random_index_shuffle",
        "//machina/core/kernels/data:random_seed_ops",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
    ],
)

tf_kernel_library(
    name = "group_by_reducer_dataset_op",
    srcs = ["group_by_reducer_dataset_op.cc"],
    deps = [
        "//machina/core:core_cpu_internal",
        "//machina/core:dataset_ops_op_lib",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core/data:captured_function",
        "//machina/core/data:dataset_utils",
    ],
)

tf_kernel_library(
    name = "group_by_window_dataset_op",
    srcs = ["group_by_window_dataset_op.cc"],
    deps = [
        "//machina/core:core_cpu_internal",
        "//machina/core:dataset_ops_op_lib",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core/data:captured_function",
        "//machina/core/data:dataset_utils",
        "//machina/core/kernels/data:window_dataset",
    ],
)

tf_kernel_library(
    name = "ignore_errors_dataset_op",
    srcs = ["ignore_errors_dataset_op.cc"],
    deps = [
        "//machina/core:experimental_dataset_ops_op_lib",
        "//machina/core:framework",
        "//machina/core:framework_internal",
        "//machina/core/platform:logging",
        "@eigen_archive//:eigen3",
    ],
)

tf_kernel_library(
    name = "index_flat_map_dataset_op",
    srcs = ["index_flat_map_dataset_op.cc"],
    deps = [
        "//machina/core:core_cpu_base",
        "//machina/core:framework",
        "//machina/core:framework_types_hdr",
        "//machina/core/data:captured_function",
        "//machina/core/data:dataset_utils",
        "//machina/core/data:name_utils",
        "//machina/core/framework:dataset_options_proto_cc",
        "//machina/core/framework:op_requires",
        "//machina/core/framework:types_proto_cc",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:statusor",
    ],
)

tf_kernel_library(
    name = "save_dataset_op",
    srcs = ["save_dataset_op.cc"],
    hdrs = ["save_dataset_op.h"],
    deps = [
        "//machina/core:experimental_dataset_ops_op_lib",
        "//machina/core:framework",
        "//machina/core:framework_internal",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core/data:captured_function",
        "//machina/core/data:dataset_utils",
        "//machina/core/data:hash_utils",
        "//machina/core/data:name_utils",
        "//machina/core/data:root_dataset",
        "//machina/core/data:snapshot_utils",
        "//machina/core/framework:op_requires",
        "//machina/core/kernels/data:iterator_ops",
    ],
)

tf_cc_test(
    name = "save_dataset_op_test",
    size = "small",
    srcs = ["save_dataset_op_test.cc"],
    deps = [
        ":save_dataset_op",
        "//machina/core:experimental_dataset_ops_op_lib",
        "//machina/core:framework",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core/data:captured_function",
        "//machina/core/data:dataset_test_base",
        "//machina/core/data:dataset_utils",
        "//machina/core/data:name_utils",
        "//machina/core/data:serialization_utils",
        "//machina/core/framework:function_testlib",
        "//machina/core/framework:types_proto_cc",
        "//machina/core/kernels:cwise_op",
        "//machina/core/kernels:function_ops",
        "//machina/core/kernels/data:range_dataset_op",
    ],
)

tf_kernel_library(
    name = "load_dataset_op",
    srcs = ["load_dataset_op.cc"],
    hdrs = ["load_dataset_op.h"],
    deps = [
        "//machina/core:experimental_dataset_ops_op_lib",
        "//machina/core:framework",
        "//machina/core:framework_internal",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core/data:captured_function",
        "//machina/core/data:name_utils",
        "//machina/core/data:snapshot_utils",
        "//machina/core/data:utils",
        "//machina/core/framework:op_requires",
    ],
)

tf_kernel_library(
    name = "list_dataset_op",
    srcs = ["list_dataset_op.cc"],
    hdrs = ["list_dataset_op.h"],
    deps = [
        "//machina/core:dataset_ops_op_lib",
        "//machina/core:framework",
        "//machina/core:graph",
        "//machina/core/data:dataset_utils",
        "//machina/core/data:global_shuffle_utils",
        "//machina/core/data:name_utils",
        "//machina/core/data:split_utils",
        "@com_google_absl//absl/status",
    ],
)

tf_cc_test(
    name = "list_dataset_op_test",
    size = "small",
    srcs = ["list_dataset_op_test.cc"],
    deps = [
        ":list_dataset_op",
        "//machina/core:test_main",
        "//machina/core/data:dataset_test_base",
        "//machina/core/data:dataset_utils",
        "//machina/core/data:serialization_utils",
    ],
)

tf_kernel_library(
    name = "lmdb_dataset_op",
    srcs = ["lmdb_dataset_op.cc"],
    hdrs = ["lmdb_dataset_op.h"],
    deps = [
        "//machina/core:experimental_dataset_ops_op_lib",
        "//machina/core:framework",
        "//machina/core:lib",
        "@eigen_archive//:eigen3",
    ],
)

tf_kernel_library(
    name = "lookup_ops",
    srcs = ["lookup_ops.cc"],
    deps = [
        "//machina/core:core_cpu",
        "//machina/core:core_cpu_internal",
        "//machina/core:framework",
        "//machina/core:framework_internal",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core:protos_all_cc",
        "//machina/core/data:root_dataset",
        "//machina/core/kernels:function_ops",
        "//machina/core/kernels:initializable_lookup_table",
        "//machina/core/kernels:lookup_util",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@eigen_archive//:eigen3",
    ],
)

tf_kernel_library(
    name = "map_and_batch_dataset_op",
    srcs = ["map_and_batch_dataset_op.cc"],
    hdrs = ["map_and_batch_dataset_op.h"],
    deps = [
        "//machina/core:array_ops_op_lib",
        "//machina/core:core_cpu_internal",
        "//machina/core:dataset_ops_op_lib",
        "//machina/core:framework",
        "//machina/core:framework_internal",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core:nn_ops_op_lib",
        "//machina/core/data:captured_function",
        "//machina/core/data:dataset_utils",
        "//machina/core/data:name_utils",
        "//machina/core/data:stats_utils",
        "//machina/core/kernels:inplace_ops",
        "//machina/core/profiler/lib:traceme",
        "//machina/core/profiler/lib:traceme_encode",
    ],
)

tf_cc_test(
    name = "map_and_batch_dataset_op_test",
    size = "small",
    srcs = ["map_and_batch_dataset_op_test.cc"],
    deps = [
        ":map_and_batch_dataset_op",
        "//machina/core:experimental_dataset_ops_op_lib",
        "//machina/core:framework",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core:testlib",
        "//machina/core/data:dataset_test_base",
        "//machina/core/kernels:cwise_op",
        "//machina/core/kernels/data:range_dataset_op",
    ],
)

tf_kernel_library(
    name = "matching_files_dataset_op",
    srcs = ["matching_files_dataset_op.cc"],
    deps = [
        "//machina/core:experimental_dataset_ops_op_lib",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:lib_internal",
    ],
)

tf_kernel_library(
    name = "non_serializable_dataset_op",
    srcs = ["non_serializable_dataset_op.cc"],
    deps = [
        "//machina/core:experimental_dataset_ops_op_lib",
        "//machina/core:framework",
        "@eigen_archive//:eigen3",
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
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core/data:captured_function",
        "//machina/core/data:dataset_utils",
        "//machina/core/data:name_utils",
        "//machina/core/profiler/lib:traceme",
        "//machina/core/profiler/lib:traceme_encode",
    ],
)

tf_cc_test(
    name = "parallel_interleave_dataset_op_test",
    size = "small",
    srcs = ["parallel_interleave_dataset_op_test.cc"],
    deps = [
        ":parallel_interleave_dataset_op",
        "//machina/core:experimental_dataset_ops_op_lib",
        "//machina/core:framework",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core:testlib",
        "//machina/core/data:dataset_test_base",
        "//machina/core/kernels/data:tensor_slice_dataset_op",
    ],
)

tf_kernel_library(
    name = "parse_example_dataset_op",
    srcs = ["parse_example_dataset_op.cc"],
    deps = [
        "//machina/core:core_cpu_internal",
        "//machina/core:framework",
        "//machina/core:functional_ops_op_lib",
        "//machina/core:lib",
        "//machina/core/data:dataset_utils",
        "//machina/core/data:name_utils",
        "//machina/core/data:stats_utils",
        "//machina/core/kernels:ragged_tensor_variant",
        "//machina/core/kernels/data:parallel_map_dataset_op",
        "//machina/core/profiler/lib:traceme",
        "//machina/core/profiler/lib:traceme_encode",
    ],
)

tf_kernel_library(
    name = "prefetching_kernels",
    srcs = ["prefetching_kernels.cc"],
    deps = [
        "//machina/core:core_cpu_internal",
        "//machina/core:experimental_dataset_ops_op_lib",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:lib_internal",
    ],
)

tf_kernel_library(
    name = "random_access_ops",
    srcs = ["random_access_ops.cc"],
    hdrs = ["random_access_ops.h"],
    deps = [
        "//machina/core:experimental_dataset_ops_op_lib",
        "//machina/core:framework",
        "//machina/core:framework_internal",
        "//machina/core:lib",
        "//machina/core/data:finalization_utils",
        "//machina/core/data:unbounded_thread_pool",
        "//machina/core/kernels/data:iterator_ops",
        "//machina/core/platform",
    ],
)

tf_kernel_library(
    name = "random_dataset_op",
    srcs = ["random_dataset_op.cc"],
    hdrs = ["random_dataset_op.h"],
    deps = [
        "//machina/core:dataset_ops_op_lib",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core/data:dataset_utils",
        "//machina/core/data:name_utils",
        "//machina/core/data:split_utils",
        "//machina/core/kernels/data:random_seed_ops",
    ],
)

tf_cc_test(
    name = "random_dataset_op_test",
    size = "small",
    srcs = ["random_dataset_op_test.cc"],
    deps = [
        ":random_dataset_op",
        "//machina/core:experimental_dataset_ops_op_lib",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core:testlib",
        "//machina/core/data:dataset_test_base",
        "//machina/core/kernels/data:random_seed_ops",
        "@eigen_archive//:eigen3",
    ],
)

tf_kernel_library(
    name = "rebatch_dataset_op",
    srcs = ["rebatch_dataset_op.cc"],
    deps = [
        "//machina/core:core_cpu_internal",
        "//machina/core:dataset_ops_op_lib",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core:protos_all_cc",
        "//machina/core/data:name_utils",
        "@com_google_absl//absl/strings",
    ],
)

tf_kernel_library(
    name = "sampling_dataset_op",
    srcs = ["sampling_dataset_op.cc"],
    hdrs = ["sampling_dataset_op.h"],
    deps = [
        "//machina/core:experimental_dataset_ops_op_lib",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core/data:dataset_utils",
        "//machina/core/data:name_utils",
    ],
)

tf_cc_test(
    name = "sampling_dataset_op_test",
    size = "small",
    srcs = ["sampling_dataset_op_test.cc"],
    deps = [
        ":sampling_dataset_op",
        "//machina/core:experimental_dataset_ops_op_lib",
        "//machina/core:framework",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core:testlib",
        "//machina/core/data:dataset_test_base",
        "//machina/core/kernels/data:range_dataset_op",
        "@eigen_archive//:eigen3",
    ],
)

tf_kernel_library(
    name = "scan_dataset_op",
    srcs = ["scan_dataset_op.cc"],
    deps = [
        "//machina/core:core_cpu_internal",
        "//machina/core:dataset_ops_op_lib",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core/data:captured_function",
        "//machina/core/data:dataset_utils",
    ],
)

tf_kernel_library(
    name = "set_stats_aggregator_dataset_op",
    srcs = ["set_stats_aggregator_dataset_op.cc"],
    deps = [
        "//machina/core:core_cpu_lib",
        "//machina/core:framework",
        "//machina/core:lib_internal",
        "//machina/core/data:stats_utils",
    ],
)

tf_kernel_library(
    name = "sleep_dataset_op",
    srcs = ["sleep_dataset_op.cc"],
    deps = [
        "//machina/core:experimental_dataset_ops_op_lib",
        "//machina/core:framework",
        "//machina/core/data:dataset_utils",
        "//machina/core/framework:dataset_options_proto_cc",
        "@com_google_absl//absl/status",
    ],
)

tf_kernel_library(
    name = "sliding_window_dataset_op",
    srcs = ["sliding_window_dataset_op.cc"],
    deps = [
        "//machina/core:dataset_ops_op_lib",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core/framework:attr_value_proto_cc",
        "//machina/core/framework:dataset_options_proto_cc",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
    ],
)

tf_kernel_library(
    name = "snapshot_dataset_op",
    srcs = ["snapshot_dataset_op.cc"],
    hdrs = ["snapshot_dataset_op.h"],
    deps = [
        "//machina/core:core_cpu_internal",
        "//machina/core:dataset_ops_op_lib",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core:protos_all_cc",
        "//machina/core/data:captured_function",
        "//machina/core/data:dataset_utils",
        "//machina/core/data:hash_utils",
        "//machina/core/data:name_utils",
        "//machina/core/data:serialization_utils",
        "//machina/core/data:snapshot_utils",
        "//machina/core/framework:op_requires",
        "//machina/core/platform:platform_port",
        "//machina/core/profiler/lib:traceme",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
    ],
)

tf_kernel_library(
    name = "sql_dataset_op",
    srcs = [
        "sql_dataset_op.cc",
    ],
    deps = [
        "//machina/core:experimental_dataset_ops_op_lib",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core/framework:types_proto_cc",
        "//machina/core/kernels/data/experimental/sql",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
    ],
)

tf_kernel_library(
    name = "stats_aggregator_ops",
    srcs = ["stats_aggregator_ops.cc"],
    deps = [
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core:protos_all_cc",
        "//machina/core/kernels:summary_interface",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:span",
    ],
)

tf_kernel_library(
    name = "stats_dataset_ops",
    srcs = ["stats_dataset_ops.cc"],
    deps = [
        "//machina/core:dataset_ops_op_lib",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core:protos_all_cc",
        "@com_google_absl//absl/status",
    ],
)

tf_kernel_library(
    name = "take_while_dataset_op",
    srcs = ["take_while_dataset_op.cc"],
    deps = [
        "//machina/core:core_cpu_internal",
        "//machina/core:dataset_ops_op_lib",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core/data:captured_function",
        "//machina/core/data:dataset_utils",
        "//machina/core/framework:attr_value_proto_cc",
        "//machina/core/framework:dataset_options_proto_cc",
        "//machina/core/framework:types_proto_cc",
        "@com_google_absl//absl/status",
    ],
)

tf_kernel_library(
    name = "threadpool_dataset_op",
    srcs = ["threadpool_dataset_op.cc"],
    hdrs = ["threadpool_dataset_op.h"],
    deps = [
        "//machina/core:experimental_dataset_ops_op_lib",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core/data:dataset_utils",
        "//machina/core/framework:dataset_options_proto_cc",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@eigen_archive//:eigen3",
    ],
)

tf_kernel_library(
    name = "to_tf_record_op",
    srcs = ["to_tf_record_op.cc"],
    deps = [
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core/data:dataset_utils",
        "//machina/core/data:root_dataset",
        "//machina/core/framework:types_proto_cc",
        "//machina/core/kernels:ops_util",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:string_view",
    ],
)

tf_kernel_library(
    name = "unbatch_dataset_op",
    srcs = ["unbatch_dataset_op.cc"],
    deps = [
        "//machina/core:dataset_ops_op_lib",
        "//machina/core:framework",
        "//machina/core:framework_internal",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core/framework:dataset_options_proto_cc",
        "//machina/core/framework:types_proto_cc",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:mutex",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:strcat",
        "@local_tsl//tsl/platform:thread_annotations",
    ],
)

tf_kernel_library(
    name = "unique_dataset_op",
    srcs = ["unique_dataset_op.cc"],
    hdrs = ["unique_dataset_op.h"],
    deps = [
        "//machina/core:experimental_dataset_ops_op_lib",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core/framework:types_proto_cc",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@eigen_archive//:eigen3",
    ],
)

tf_kernel_library(
    name = "weighted_flat_map_dataset_op",
    srcs = ["weighted_flat_map_dataset_op.cc"],
    deps = [
        "//machina/core:dataset_ops_op_lib",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core/data:captured_function",
        "//machina/core/data:name_utils",
        "//machina/core/framework:dataset_options_proto_cc",
        "//machina/core/framework:types_proto_cc",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
    ],
)

tf_cc_test(
    name = "unique_dataset_op_test",
    size = "small",
    srcs = ["unique_dataset_op_test.cc"],
    deps = [
        ":unique_dataset_op",
        "//machina/core:experimental_dataset_ops_op_lib",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core:testlib",
        "//machina/core/data:dataset_test_base",
        "//machina/core/framework:types_proto_cc",
        "//machina/core/kernels/data:tensor_slice_dataset_op",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest",
        "@eigen_archive//:eigen3",
    ],
)

tf_kernel_library(
    # data service kernels depend on GRPC, so we package them separately
    # so that downstream rules can avoid depending on GRPC.
    name = "data_service_kernels",
    deps = [
        ":data_service_dataset_op",
        ":data_service_ops",
        ":distributed_save_op",
    ],
)

tf_kernel_library(
    name = "experimental",
    deps = [
        ":assert_cardinality_dataset_op",
        ":assert_next_dataset_op",
        ":assert_prev_dataset_op",
        ":check_pinned_op",
        ":choose_fastest_branch_dataset_op",
        ":choose_fastest_dataset_op",
        ":compression_ops",
        ":csv_dataset_op",
        ":dense_to_sparse_batch_dataset_op",
        ":directed_interleave_dataset_op",
        ":global_shuffle_dataset_op",
        ":group_by_reducer_dataset_op",
        ":group_by_window_dataset_op",
        ":ignore_errors_dataset_op",
        ":index_flat_map_dataset_op",
        ":list_dataset_op",
        ":load_dataset_op",
        ":lookup_ops",
        ":map_and_batch_dataset_op",
        ":matching_files_dataset_op",
        ":non_serializable_dataset_op",
        ":parallel_interleave_dataset_op",
        ":parse_example_dataset_op",
        ":prefetching_kernels",
        ":random_access_ops",
        ":random_dataset_op",
        ":rebatch_dataset_op",
        ":sampling_dataset_op",
        ":save_dataset_op",
        ":scan_dataset_op",
        ":set_stats_aggregator_dataset_op",
        ":sleep_dataset_op",
        ":sliding_window_dataset_op",
        ":snapshot_dataset_op",
        ":sql_dataset_op",
        ":stats_aggregator_ops",
        ":stats_dataset_ops",
        ":take_while_dataset_op",
        ":threadpool_dataset_op",
        ":to_tf_record_op",
        ":unbatch_dataset_op",
        ":unique_dataset_op",
        ":weighted_flat_map_dataset_op",
        "//machina/core/data/service/snapshot:list_snapshot_chunks_dataset_op",
        "//machina/core/data/service/snapshot:snapshot_chunk_dataset_op",
    ] + select({
        "//machina:fuchsia": [],
        "//conditions:default": [":lmdb_dataset_op"],
    }) + if_not_mobile([
        # auto_shard_dataset_op uses RewriteDataset method, which
        # is not defined on mobile platforms
        ":auto_shard_dataset_op",
        # compute_batch_size_op depends on grappler, which
        # should not be included on mobile platforms
        ":compute_batch_size_op",
    ]),
)

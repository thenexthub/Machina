# Definitions are loaded separately so that copybara can pattern match (and modify) each definition.
load("//machina:machina.default.bzl", "cuda_py_strict_test", "tf_py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    licenses = ["notice"],
)

tf_py_strict_test(
    name = "assert_cardinality_test",
    size = "small",
    srcs = ["assert_cardinality_test.py"],
    deps = [
        "//machina/python/data/experimental/ops:cardinality",
        "//machina/python/data/experimental/ops:global_shuffle_op",
        "//machina/python/data/experimental/ops:random_access",
        "//machina/python/data/kernel_tests:checkpoint_test_base",
        "//machina/python/data/kernel_tests:test_base",
        "//machina/python/data/ops:dataset_ops",
        "//machina/python/data/ops:options",
        "//machina/python/framework:combinations",
        "//machina/python/framework:errors",
        "//machina/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
    name = "assert_next_test",
    size = "small",
    srcs = ["assert_next_test.py"],
    deps = [
        "//machina/python/data/experimental/ops:testing",
        "//machina/python/data/kernel_tests:test_base",
        "//machina/python/data/ops:dataset_ops",
        "//machina/python/data/ops:options",
        "//machina/python/framework:combinations",
        "//machina/python/framework:errors",
        "//machina/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
    name = "assert_prev_test",
    size = "small",
    srcs = ["assert_prev_test.py"],
    deps = [
        "//machina/python/data/experimental/ops:testing",
        "//machina/python/data/kernel_tests:test_base",
        "//machina/python/data/ops:dataset_ops",
        "//machina/python/data/ops:options",
        "//machina/python/framework:combinations",
        "//machina/python/framework:errors",
        "//machina/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
    name = "auto_shard_dataset_test",
    size = "medium",
    srcs = ["auto_shard_dataset_test.py"],
    shard_count = 4,
    deps = [
        "//machina/core:protos_all_py",
        "//machina/python/data/experimental/ops:cardinality",
        "//machina/python/data/experimental/ops:distribute",
        "//machina/python/data/experimental/ops:global_shuffle_op",
        "//machina/python/data/experimental/ops:interleave_ops",
        "//machina/python/data/experimental/ops:readers",
        "//machina/python/data/experimental/ops:testing",
        "//machina/python/data/experimental/ops:unique",
        "//machina/python/data/kernel_tests:checkpoint_test_base",
        "//machina/python/data/kernel_tests:test_base",
        "//machina/python/data/kernel_tests:tf_record_test_base",
        "//machina/python/data/ops:dataset_ops",
        "//machina/python/data/ops:options",
        "//machina/python/data/ops:readers",
        "//machina/python/framework:combinations",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/lib/io:python_io",
        "//machina/python/ops:parsing_ops",
        "//machina/python/ops:string_ops",
        "//machina/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
    name = "compression_ops_test",
    size = "small",
    srcs = ["compression_ops_test.py"],
    deps = [
        "//machina/python/data/experimental/ops:compression_ops",
        "//machina/python/data/kernel_tests:test_base",
        "//machina/python/data/ops:dataset_ops",
        "//machina/python/data/util:structure",
        "//machina/python/eager:context",
        "//machina/python/framework:combinations",
        "//machina/python/framework:errors",
        "//machina/python/framework:sparse_tensor",
        "//machina/python/ops/ragged:ragged_factory_ops",
        "//machina/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_strict_test(
    name = "copy_to_device_test",
    size = "small",
    srcs = ["copy_to_device_test.py"],
    deps = [
        "//machina/core:protos_all_py",
        "//machina/python/data/experimental/ops:prefetching_ops",
        "//machina/python/data/experimental/ops:testing",
        "//machina/python/data/kernel_tests:test_base",
        "//machina/python/data/ops:dataset_ops",
        "//machina/python/data/ops:iterator_ops",
        "//machina/python/data/ops:options",
        "//machina/python/data/util:structure",
        "//machina/python/framework:combinations",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/framework:ops",
        "//machina/python/framework:sparse_tensor",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:math_ops",
        "//machina/python/platform:client_testlib",
        "//machina/python/util:compat",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
    name = "csv_dataset_test",
    size = "medium",
    srcs = ["csv_dataset_test.py"],
    shard_count = 8,
    deps = [
        "//machina/python/data/experimental/ops:error_ops",
        "//machina/python/data/experimental/ops:readers",
        "//machina/python/data/kernel_tests:checkpoint_test_base",
        "//machina/python/data/kernel_tests:test_base",
        "//machina/python/data/ops:readers",
        "//machina/python/eager:context",
        "//machina/python/framework:combinations",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/ops:parsing_ops",
        "//machina/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
    name = "dense_to_sparse_batch_test",
    size = "medium",
    srcs = ["dense_to_sparse_batch_test.py"],
    deps = [
        "//machina/python/data/experimental/ops:batching",
        "//machina/python/data/kernel_tests:checkpoint_test_base",
        "//machina/python/data/kernel_tests:test_base",
        "//machina/python/data/ops:dataset_ops",
        "//machina/python/framework:combinations",
        "//machina/python/framework:errors",
        "//machina/python/ops:array_ops",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
    name = "from_list_test",
    size = "small",
    timeout = "moderate",
    srcs = ["from_list_test.py"],
    shard_count = 4,
    deps = [
        "//machina/python/data/experimental/ops:from_list",
        "//machina/python/data/experimental/ops:global_shuffle_op",
        "//machina/python/data/experimental/ops:random_access",
        "//machina/python/data/kernel_tests:checkpoint_test_base",
        "//machina/python/data/kernel_tests:test_base",
        "//machina/python/data/ops:dataset_ops",
        "//machina/python/data/ops:options",
        "//machina/python/framework:combinations",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/framework:tensor_shape",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
    name = "global_shuffle_test",
    srcs = ["global_shuffle_test.py"],
    shard_count = 8,
    deps = [
        "//machina/python/data/experimental/ops:global_shuffle_op",
        "//machina/python/data/kernel_tests:checkpoint_test_base",
        "//machina/python/data/kernel_tests:test_base",
        "//machina/python/data/ops:dataset_ops",
        "//machina/python/data/ops:options",
        "//machina/python/framework:combinations",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/framework:random_seed",
        "//machina/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
    name = "group_by_reducer_test",
    size = "small",
    srcs = ["group_by_reducer_test.py"],
    shard_count = 12,
    deps = [
        "//machina/python/data/experimental/ops:grouping",
        "//machina/python/data/kernel_tests:checkpoint_test_base",
        "//machina/python/data/kernel_tests:test_base",
        "//machina/python/data/ops:dataset_ops",
        "//machina/python/framework:combinations",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/framework:sparse_tensor",
        "//machina/python/framework:tensor_shape",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
    name = "index_flat_map_test",
    srcs = ["index_flat_map_test.py"],
    shard_count = 8,
    deps = [
        "//machina/python/data/experimental/ops:global_shuffle_op",
        "//machina/python/data/experimental/ops:index_flat_map_op",
        "//machina/python/data/kernel_tests:checkpoint_test_base",
        "//machina/python/data/kernel_tests:test_base",
        "//machina/python/data/ops:dataset_ops",
        "//machina/python/data/ops:options",
        "//machina/python/framework:combinations",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/framework:tensor",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops/ragged:ragged_string_ops",
        "//machina/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
    name = "index_shuffle_test",
    size = "large",
    srcs = ["index_shuffle_test.py"],
    shard_count = 12,
    deps = [
        "//machina/python/data/experimental/ops:shuffle_ops",
        "//machina/python/data/kernel_tests:checkpoint_test_base",
        "//machina/python/data/kernel_tests:test_base",
        "//machina/python/data/ops:dataset_ops",
        "//machina/python/data/ops:options",
        "//machina/python/framework:combinations",
        "//machina/python/framework:dtypes",
        "//machina/python/ops:array_ops",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
    name = "io_test",
    size = "medium",
    srcs = ["io_test.py"],
    tags = [
        "notap",  # b/272281090
    ],
    deps = [
        "//machina/python/data/experimental/ops:io",
        "//machina/python/data/kernel_tests:checkpoint_test_base",
        "//machina/python/data/kernel_tests:test_base",
        "//machina/python/data/ops:dataset_ops",
        "//machina/python/eager:def_function",
        "//machina/python/framework:combinations",
        "//machina/python/ops:variables",
        "//machina/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
    name = "lookup_ops_test",
    size = "small",
    srcs = ["lookup_ops_test.py"],
    deps = [
        "//machina/python:tf2",
        "//machina/python/data/experimental/ops:lookup_ops",
        "//machina/python/data/ops:dataset_ops",
        "//machina/python/data/ops:readers",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/ops:lookup_ops",
        "//machina/python/ops:string_ops",
        "//machina/python/ops:variables",
        "//machina/python/platform:client_testlib",
    ],
)

tf_py_strict_test(
    name = "make_batched_features_dataset_test",
    size = "small",
    srcs = ["make_batched_features_dataset_test.py"],
    shard_count = 6,
    deps = [
        "//machina/python/data/experimental/ops:readers",
        "//machina/python/data/kernel_tests:test_base",
        "//machina/python/data/kernel_tests:tf_record_test_base",
        "//machina/python/data/ops:dataset_ops",
        "//machina/python/data/ops:readers",
        "//machina/python/data/util:nest",
        "//machina/python/framework:combinations",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/framework:tensor",
        "//machina/python/ops:io_ops",
        "//machina/python/ops:parsing_ops",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
    name = "make_csv_dataset_test",
    size = "small",
    srcs = ["make_csv_dataset_test.py"],
    deps = [
        "//machina/python/data/experimental/ops:readers",
        "//machina/python/data/kernel_tests:test_base",
        "//machina/python/data/ops:dataset_ops",
        "//machina/python/data/util:nest",
        "//machina/python/framework:combinations",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
    name = "make_saveable_from_iterator_test",
    size = "small",
    srcs = ["make_saveable_from_iterator_test.py"],
    deps = [
        "//machina/python/data/experimental/ops:iterator_ops",
        "//machina/python/data/ops:dataset_ops",
        "//machina/python/framework:combinations",
        "//machina/python/framework:errors",
        "//machina/python/framework:ops",
        "//machina/python/platform:client_testlib",
        "//machina/python/training:saver",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
    name = "make_tf_record_dataset_test",
    size = "medium",
    srcs = ["make_tf_record_dataset_test.py"],
    deps = [
        "//machina/python/data/experimental/ops:readers",
        "//machina/python/data/kernel_tests:test_base",
        "//machina/python/data/kernel_tests:tf_record_test_base",
        "//machina/python/data/ops:dataset_ops",
        "//machina/python/data/util:nest",
        "//machina/python/framework:combinations",
        "//machina/python/framework:errors",
        "//machina/python/ops:string_ops",
        "//machina/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
    name = "map_and_batch_test",
    size = "medium",
    srcs = ["map_and_batch_test.py"],
    shard_count = 4,
    deps = [
        "//machina/python:pywrap_sanitizers",
        "//machina/python/checkpoint",
        "//machina/python/checkpoint:checkpoint_management",
        "//machina/python/data/experimental/ops:batching",
        "//machina/python/data/kernel_tests:checkpoint_test_base",
        "//machina/python/data/kernel_tests:test_base",
        "//machina/python/data/ops:dataset_ops",
        "//machina/python/data/ops:options",
        "//machina/python/eager:context",
        "//machina/python/framework:combinations",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/framework:sparse_tensor",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:cond",
        "//machina/python/ops:control_flow_util",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:script_ops",
        "//machina/python/ops:variables",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
    name = "map_defun_op_test",
    size = "small",
    srcs = ["map_defun_op_test.py"],
    deps = [
        "//machina/python/client:session",
        "//machina/python/data/experimental/ops:map_defun",
        "//machina/python/data/kernel_tests:test_base",
        "//machina/python/eager:def_function",
        "//machina/python/framework:combinations",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/framework:ops",
        "//machina/python/framework:sparse_tensor",
        "//machina/python/framework:tensor_spec",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:array_ops_stack",
        "//machina/python/ops:check_ops",
        "//machina/python/ops:data_flow_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:sparse_ops",
        "//machina/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
    name = "matching_files_dataset_test",
    size = "small",
    srcs = ["matching_files_dataset_test.py"],
    deps = [
        "//machina/python/data/experimental/ops:matching_files",
        "//machina/python/data/kernel_tests:checkpoint_test_base",
        "//machina/python/data/kernel_tests:test_base",
        "//machina/python/framework:combinations",
        "//machina/python/framework:errors",
        "//machina/python/platform:client_testlib",
        "//machina/python/util:compat",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
    name = "model_dataset_test",
    size = "small",
    srcs = ["model_dataset_test.py"],
    deps = [
        "//machina/python/data/experimental/ops:testing",
        "//machina/python/data/kernel_tests:test_base",
        "//machina/python/data/ops:dataset_ops",
        "//machina/python/data/ops:options",
        "//machina/python/framework:combinations",
        "//machina/python/framework:errors",
        "//machina/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
    name = "non_serializable_test",
    size = "small",
    srcs = ["non_serializable_test.py"],
    deps = [
        "//machina/python/data/experimental/ops:testing",
        "//machina/python/data/kernel_tests:test_base",
        "//machina/python/data/ops:dataset_ops",
        "//machina/python/data/ops:options",
        "//machina/python/framework:combinations",
        "//machina/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
    name = "pad_to_cardinality_test",
    srcs = ["pad_to_cardinality_test.py"],
    deps = [
        "//machina/python/data/experimental/ops:pad_to_cardinality",
        "//machina/python/data/kernel_tests:test_base",
        "//machina/python/data/ops:dataset_ops",
        "//machina/python/framework:combinations",
        "//machina/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
    name = "parallel_interleave_test",
    size = "medium",
    srcs = ["parallel_interleave_test.py"],
    shard_count = 8,
    deps = [
        "//machina/python/data/experimental/ops:interleave_ops",
        "//machina/python/data/experimental/ops:testing",
        "//machina/python/data/kernel_tests:checkpoint_test_base",
        "//machina/python/data/kernel_tests:test_base",
        "//machina/python/data/ops:dataset_ops",
        "//machina/python/data/ops:options",
        "//machina/python/framework:combinations",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/framework:sparse_tensor",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:script_ops",
        "//machina/python/ops:sparse_ops",
        "//machina/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
    name = "parse_example_dataset_test",
    size = "medium",
    srcs = ["parse_example_dataset_test.py"],
    shard_count = 4,
    deps = [
        "//machina/core:protos_all_py",
        "//machina/python/data/experimental/ops:parsing_ops",
        "//machina/python/data/kernel_tests:checkpoint_test_base",
        "//machina/python/data/kernel_tests:test_base",
        "//machina/python/data/kernel_tests:tf_record_test_base",
        "//machina/python/data/ops:dataset_ops",
        "//machina/python/data/ops:options",
        "//machina/python/eager:context",
        "//machina/python/framework:combinations",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/framework:ops",
        "//machina/python/framework:sparse_tensor",
        "//machina/python/framework:tensor",
        "//machina/python/ops:parsing_ops",
        "//machina/python/ops/ragged:ragged_factory_ops",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_strict_test(
    name = "prefetch_to_device_test",
    size = "small",
    srcs = ["prefetch_to_device_test.py"],
    deps = [
        "//machina/core:protos_all_py",
        "//machina/python/data/experimental/ops:prefetching_ops",
        "//machina/python/data/kernel_tests:test_base",
        "//machina/python/data/ops:dataset_ops",
        "//machina/python/data/util:structure",
        "//machina/python/framework:combinations",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/framework:ops",
        "//machina/python/framework:sparse_tensor",
        "//machina/python/framework:test_lib",
        "//machina/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
    name = "prefetch_with_slack_test",
    size = "small",
    srcs = ["prefetch_with_slack_test.py"],
    deps = [
        "//machina/python/data/kernel_tests:test_base",
        "//machina/python/data/ops:dataset_ops",
        "//machina/python/data/ops:multi_device_iterator_ops",
        "//machina/python/data/ops:options",
        "//machina/python/framework:combinations",
        "//machina/python/framework:errors",
        "//machina/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
    name = "rebatch_dataset_test",
    size = "medium",
    srcs = ["rebatch_dataset_test.py"],
    shard_count = 4,
    deps = [
        "//machina/python/data/experimental/ops:distribute",
        "//machina/python/data/kernel_tests:checkpoint_test_base",
        "//machina/python/data/kernel_tests:test_base",
        "//machina/python/data/ops:dataset_ops",
        "//machina/python/data/util:nest",
        "//machina/python/framework:combinations",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:tensor_shape",
        "//machina/python/framework:tensor_util",
        "//machina/python/ops:image_ops",
        "//machina/python/ops/ragged:ragged_tensor",
        "//machina/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
    name = "replicate_test",
    size = "medium",
    srcs = ["replicate_test.py"],
    grpc_enabled = True,
    tags = ["no_oss"],
    deps = [
        "//machina/core:protos_all_py",
        "//machina/python:pywrap_tfe",
        "//machina/python/client:session",
        "//machina/python/data/experimental/ops:distribute",
        "//machina/python/data/kernel_tests:test_base",
        "//machina/python/data/ops:dataset_ops",
        "//machina/python/data/ops:options",
        "//machina/python/eager:context",
        "//machina/python/framework:combinations",
        "//machina/python/framework:config",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/framework:ops",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:random_ops",
        "//machina/python/ops:variable_scope",
        "//machina/python/platform:client_testlib",
        "//machina/python/training:server_lib",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
    name = "shuffle_and_repeat_test",
    size = "medium",
    srcs = ["shuffle_and_repeat_test.py"],
    deps = [
        "//machina/python/data/experimental/ops:shuffle_ops",
        "//machina/python/data/kernel_tests:checkpoint_test_base",
        "//machina/python/data/kernel_tests:test_base",
        "//machina/python/data/ops:dataset_ops",
        "//machina/python/framework:combinations",
        "//machina/python/framework:errors",
        "//machina/python/framework:random_seed",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
    name = "sleep_test",
    size = "small",
    srcs = ["sleep_test.py"],
    deps = [
        "//machina/python/data/experimental/ops:testing",
        "//machina/python/data/kernel_tests:test_base",
        "//machina/python/data/ops:dataset_ops",
        "//machina/python/framework:combinations",
        "//machina/python/framework:errors",
        "//machina/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
    name = "sql_dataset_test",
    size = "medium",
    srcs = ["sql_dataset_test.py"],
    shard_count = 4,
    deps = [
        "//machina/python/data/experimental/ops:readers",
        "//machina/python/data/kernel_tests:checkpoint_test_base",
        "//machina/python/data/kernel_tests:test_base",
        "//machina/python/framework:combinations",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/ops:array_ops",
        "//machina/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
        "@org_sqlite//:python",
    ],
)

tf_py_strict_test(
    name = "tf_record_writer_test",
    size = "medium",
    srcs = ["tf_record_writer_test.py"],
    deps = [
        "//machina/python/data/experimental/ops:grouping",
        "//machina/python/data/experimental/ops:writers",
        "//machina/python/data/kernel_tests:test_base",
        "//machina/python/data/ops:dataset_ops",
        "//machina/python/data/ops:readers",
        "//machina/python/eager:def_function",
        "//machina/python/framework:combinations",
        "//machina/python/framework:dtypes",
        "//machina/python/lib/io:python_io",
        "//machina/python/lib/io:tf_record",
        "//machina/python/ops:string_ops",
        "//machina/python/platform:client_testlib",
        "//machina/python/util:compat",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
    name = "variant_test",
    size = "small",
    srcs = ["variant_test.py"],
    deps = [
        "//machina/python/data/kernel_tests:test_base",
        "//machina/python/data/ops:dataset_ops",
        "//machina/python/framework:combinations",
        "//machina/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
    name = "weighted_flat_map_test",
    srcs = ["weighted_flat_map_test.py"],
    deps = [
        "//machina/python/data/experimental/ops:global_shuffle_op",
        "//machina/python/data/experimental/ops:weighted_flat_map_op",
        "//machina/python/data/kernel_tests:checkpoint_test_base",
        "//machina/python/data/kernel_tests:test_base",
        "//machina/python/data/ops:dataset_ops",
        "//machina/python/data/ops:options",
        "//machina/python/framework:combinations",
        "//machina/python/framework:errors",
        "//machina/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_strict_test(
    name = "wrap_unwrap_test",
    size = "small",
    srcs = ["wrap_unwrap_test.py"],
    deps = [
        "//machina/python/data/kernel_tests:test_base",
        "//machina/python/data/ops:dataset_ops",
        "//machina/python/framework:combinations",
        "//machina/python/framework:ops",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:dataset_ops_gen",
        "//machina/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
    name = "iterator_model_ops_test",
    size = "small",
    srcs = ["iterator_model_ops_test.py"],
    deps = [
        "//machina/python/data/experimental/ops:cardinality",
        "//machina/python/data/experimental/ops:iterator_model_ops",
        "//machina/python/data/experimental/ops:iterator_ops",
        "//machina/python/data/kernel_tests:checkpoint_test_base",
        "//machina/python/data/kernel_tests:test_base",
        "//machina/python/data/ops:dataset_ops",
        "//machina/python/data/ops:options",
        "//machina/python/framework:combinations",
        "//machina/python/framework:errors",
        "//machina/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

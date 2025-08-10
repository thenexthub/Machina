load("//machina:strict.default.bzl", "py_strict_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//machina:internal"],
    licenses = ["notice"],
)

py_strict_library(
    name = "batching",
    srcs = ["batching.py"],
    deps = [
        "//machina/python/data/ops:dataset_ops",
        "//machina/python/data/ops:structured_function",
        "//machina/python/data/util:convert",
        "//machina/python/data/util:nest",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/framework:sparse_tensor",
        "//machina/python/framework:tensor_shape",
        "//machina/python/framework:tensor_util",
        "//machina/python/ops:experimental_dataset_ops_gen",
        "//machina/python/util:deprecation",
        "//machina/python/util:tf_export",
    ],
)

py_strict_library(
    name = "cardinality",
    srcs = ["cardinality.py"],
    deps = [
        "//machina/python/data/ops:dataset_ops",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/ops:dataset_ops_gen",
        "//machina/python/ops:experimental_dataset_ops_gen",
        "//machina/python/util:tf_export",
    ],
)

py_strict_library(
    name = "compression_ops",
    srcs = ["compression_ops.py"],
    deps = [
        "//machina/python/data/util:structure",
        "//machina/python/ops:experimental_dataset_ops_gen",
    ],
)

py_strict_library(
    name = "counter",
    srcs = ["counter.py"],
    deps = [
        "//machina/python:tf2",
        "//machina/python/compat:v2_compat",
        "//machina/python/data/ops:dataset_ops",
        "//machina/python/framework:dtypes",
        "//machina/python/util:deprecation",
        "//machina/python/util:tf_export",
    ],
)

py_strict_library(
    name = "data_service_ops",
    srcs = [
        "data_service_ops.py",
    ],
    deps = [
        ":compression_ops",
        "//machina/core:protos_all_py",
        "//machina/python:tf2",
        "//machina/python/data/experimental/service:_pywrap_server_lib",
        "//machina/python/data/experimental/service:_pywrap_utils_exp",
        "//machina/python/data/ops:dataset_ops",
        "//machina/python/data/ops:options",
        "//machina/python/data/ops:structured_function",
        "//machina/python/eager:context",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor",
        "//machina/python/framework:tensor_util",
        "//machina/python/ops:experimental_dataset_ops_gen",
        "//machina/python/ops:string_ops",
        "//machina/python/saved_model:nested_structure_coder",
        "//machina/python/util:tf_export",
    ],
)

py_strict_library(
    name = "distributed_save_op",
    srcs = [
        "distributed_save_op.py",
    ],
    visibility = ["//machina:internal"],
    deps = [
        "//machina/core:protos_all_py",
        "//machina/python/data/ops:dataset_ops",
        "//machina/python/framework:ops",
        "//machina/python/ops:experimental_dataset_ops_gen",
        "//machina/python/saved_model:nested_structure_coder",
        "//machina/python/util:tf_export",
    ],
)

py_strict_library(
    name = "distribute",
    srcs = [
        "distribute.py",
    ],
    deps = [
        "//machina/python/data/ops:dataset_ops",
        "//machina/python/data/ops:options",
        "//machina/python/data/util:nest",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor_shape",
        "//machina/python/framework:tensor_util",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:experimental_dataset_ops_gen",
        "//machina/python/types:data",
        "//machina/python/util:tf_export",
    ],
)

py_strict_library(
    name = "enumerate_ops",
    srcs = ["enumerate_ops.py"],
    deps = [
        "//machina/python/util:deprecation",
        "//machina/python/util:tf_export",
    ],
)

py_strict_library(
    name = "error_ops",
    srcs = ["error_ops.py"],
    deps = [
        "//machina/python/util:deprecation",
        "//machina/python/util:tf_export",
    ],
)

py_strict_library(
    name = "from_list",
    srcs = ["from_list.py"],
    deps = [
        "//machina/python/data/ops:dataset_ops",
        "//machina/python/data/util:nest",
        "//machina/python/data/util:structure",
        "//machina/python/ops:experimental_dataset_ops_gen",
        "//machina/python/util:tf_export",
    ],
)

py_strict_library(
    name = "get_single_element",
    srcs = ["get_single_element.py"],
    deps = [
        "//machina/python/data/ops:dataset_ops",
        "//machina/python/types:data",
        "//machina/python/util:deprecation",
        "//machina/python/util:tf_export",
    ],
)

py_strict_library(
    name = "global_shuffle_op",
    srcs = [
        "global_shuffle_op.py",
    ],
    visibility = ["//machina:internal"],
    deps = [
        "//machina/python/data/ops:dataset_ops",
        "//machina/python/data/ops:options",
        "//machina/python/data/util:random_seed",
        "//machina/python/framework:tensor",
        "//machina/python/ops:dataset_ops_gen",
        "//machina/python/ops:experimental_dataset_ops_gen",
    ],
)

py_strict_library(
    name = "grouping",
    srcs = ["grouping.py"],
    deps = [
        "//machina/python/data/ops:dataset_ops",
        "//machina/python/data/ops:structured_function",
        "//machina/python/data/util:nest",
        "//machina/python/data/util:structure",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor_spec",
        "//machina/python/ops:experimental_dataset_ops_gen",
        "//machina/python/util:deprecation",
        "//machina/python/util:tf_export",
    ],
)

py_strict_library(
    name = "index_flat_map_op",
    srcs = [
        "index_flat_map_op.py",
    ],
    deps = [
        "//machina/python/data/ops:dataset_ops",
        "//machina/python/data/ops:structured_function",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor",
        "//machina/python/framework:tensor_spec",
        "//machina/python/ops:experimental_dataset_ops_gen",
    ],
)

py_strict_library(
    name = "interleave_ops",
    srcs = ["interleave_ops.py"],
    deps = [
        "//machina/python:tf2",
        "//machina/python/compat:v2_compat",
        "//machina/python/data/ops:dataset_ops",
        "//machina/python/data/ops:readers",
        "//machina/python/util:deprecation",
        "//machina/python/util:tf_export",
    ],
)

py_strict_library(
    name = "io",
    srcs = [
        "io.py",
    ],
    deps = [
        "//machina/python/data/ops:dataset_ops",
        "//machina/python/util:deprecation",
        "//machina/python/util:tf_export",
    ],
)

py_strict_library(
    name = "iterator_ops",
    srcs = [
        "iterator_ops.py",
    ],
    deps = [
        "//machina/python/data/ops:iterator_ops",
        "//machina/python/data/ops:options",
        "//machina/python/framework:ops",
        "//machina/python/ops:proto_ops",
        "//machina/python/util:deprecation",
        "//machina/python/util:tf_export",
    ],
)

py_strict_library(
    name = "iterator_model_ops",
    srcs = [
        "iterator_model_ops.py",
    ],
    deps = [
        "//machina/core:protos_all_py",
        "//machina/python/data/ops:dataset_ops",
        "//machina/python/data/ops:iterator_ops",
        "//machina/python/eager:context",
        "//machina/python/framework:ops",
        "//machina/python/ops:experimental_dataset_ops_gen",
        "//machina/python/util:tf_export",
    ],
)

py_strict_library(
    name = "lookup_ops",
    srcs = [
        "lookup_ops.py",
    ],
    deps = [
        ":cardinality",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor",
        "//machina/python/ops:experimental_dataset_ops_gen",
        "//machina/python/ops:lookup_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/util:tf_export",
    ],
)

py_strict_library(
    name = "map_defun",
    srcs = ["map_defun.py"],
    deps = [
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor_shape",
        "//machina/python/ops:dataset_ops_gen",
    ],
)

py_strict_library(
    name = "matching_files",
    srcs = ["matching_files.py"],
    deps = [
        "//machina/python/data/ops:dataset_ops",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor_spec",
        "//machina/python/ops:experimental_dataset_ops_gen",
    ],
)

py_strict_library(
    name = "pad_to_cardinality",
    srcs = ["pad_to_cardinality.py"],
    deps = [
        "//machina/python/data/ops:dataset_ops",
        "//machina/python/eager:context",
        "//machina/python/ops:array_ops",
        "//machina/python/util:nest",
        "//machina/python/util:tf_export",
    ],
)

py_strict_library(
    name = "parsing_ops",
    srcs = ["parsing_ops.py"],
    deps = [
        "//machina/python/data/ops:dataset_ops",
        "//machina/python/data/util:structure",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:sparse_tensor",
        "//machina/python/framework:tensor_spec",
        "//machina/python/ops:experimental_dataset_ops_gen",
        "//machina/python/ops:parsing_ops",
        "//machina/python/ops/ragged:ragged_tensor",
        "//machina/python/util:deprecation",
        "//machina/python/util:tf_export",
    ],
)

py_strict_library(
    name = "prefetching_ops",
    srcs = ["prefetching_ops.py"],
    deps = [
        "//machina/python/data/ops:dataset_ops",
        "//machina/python/data/ops:iterator_ops",
        "//machina/python/data/ops:structured_function",
        "//machina/python/data/util:structure",
        "//machina/python/eager:def_function",
        "//machina/python/eager:function",
        "//machina/python/framework:device",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor_spec",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:dataset_ops_gen",
        "//machina/python/ops:experimental_dataset_ops_gen",
        "//machina/python/ops:functional_ops",
        "//machina/python/ops:resource_variable_ops",
        "//machina/python/util:tf_export",
    ],
)

py_strict_library(
    name = "random_access",
    srcs = ["random_access.py"],
    deps = [
        "//machina/python/data/util:structure",
        "//machina/python/ops:experimental_dataset_ops_gen",
        "//machina/python/util:tf_export",
    ],
)

py_strict_library(
    name = "random_ops",
    srcs = [
        "random_ops.py",
    ],
    deps = [
        "//machina/python:tf2",
        "//machina/python/compat:v2_compat",
        "//machina/python/data/ops:dataset_ops",
        "//machina/python/util:deprecation",
        "//machina/python/util:tf_export",
    ],
)

py_strict_library(
    name = "readers",
    srcs = [
        "readers.py",
    ],
    deps = [
        ":error_ops",
        ":parsing_ops",
        "//machina/python:tf2",
        "//machina/python/compat:v2_compat",
        "//machina/python/data/ops:dataset_ops",
        "//machina/python/data/ops:options",
        "//machina/python/data/ops:readers",
        "//machina/python/data/util:convert",
        "//machina/python/data/util:nest",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor_spec",
        "//machina/python/framework:tensor_util",
        "//machina/python/lib/io:file_io",
        "//machina/python/ops:experimental_dataset_ops_gen",
        "//machina/python/ops:io_ops",
        "//machina/python/platform:gfile",
        "//machina/python/util:tf_export",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "resampling",
    srcs = ["resampling.py"],
    deps = [
        "//machina/python/util:deprecation",
        "//machina/python/util:tf_export",
    ],
)

py_strict_library(
    name = "scan_ops",
    srcs = ["scan_ops.py"],
    deps = [
        "//machina/python/util:deprecation",
        "//machina/python/util:tf_export",
    ],
)

py_strict_library(
    name = "shuffle_ops",
    srcs = [
        "shuffle_ops.py",
    ],
    visibility = [
        # TODO(jsimsa): Remove non-default visibility exception when `index_shuffle` is made public.
        "//third_party/py/machina_datasets:__subpackages__",
        "//machina:internal",
    ],
    deps = [
        ":random_access",
        "//machina/python/data/ops:dataset_ops",
        "//machina/python/data/util:random_seed",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:dataset_ops_gen",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:stateless_random_ops",
        "//machina/python/util:deprecation",
        "//machina/python/util:tf_export",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "snapshot",
    srcs = [
        "snapshot.py",
    ],
    visibility = ["//machina:internal"],
    deps = [
        "//machina/python/data/ops:dataset_ops",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/framework:random_seed",
        "//machina/python/ops:experimental_dataset_ops_gen",
        "//machina/python/util:deprecation",
        "//machina/python/util:tf_export",
    ],
)

py_strict_library(
    name = "take_while_ops",
    srcs = ["take_while_ops.py"],
    deps = [
        "//machina/python/util:deprecation",
        "//machina/python/util:tf_export",
    ],
)

py_strict_library(
    name = "testing",
    srcs = ["testing.py"],
    deps = [
        "//machina/core:protos_all_py",
        "//machina/python/data/ops:dataset_ops",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/ops:experimental_dataset_ops_gen",
    ],
)

py_strict_library(
    name = "unique",
    srcs = [
        "unique.py",
    ],
    deps = [
        "//machina/python/util:deprecation",
        "//machina/python/util:tf_export",
    ],
)

py_strict_library(
    name = "weighted_flat_map_op",
    srcs = [
        "weighted_flat_map_op.py",
    ],
    deps = [
        "//machina/python/data/ops:dataset_ops",
        "//machina/python/data/util:convert",
        "//machina/python/data/util:nest",
        "//machina/python/data/util:random_seed",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor",
        "//machina/python/ops:dataset_ops_gen",
        "//machina/python/ops:experimental_dataset_ops_gen",
    ],
)

py_strict_library(
    name = "writers",
    srcs = [
        "writers.py",
    ],
    deps = [
        "//machina/python/data/ops:dataset_ops",
        "//machina/python/data/util:convert",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor_spec",
        "//machina/python/ops:experimental_dataset_ops_gen",
        "//machina/python/types:data",
        "//machina/python/util:deprecation",
        "//machina/python/util:tf_export",
    ],
)

py_strict_library(
    name = "dataset_ops",
    deps = [
        ":batching",
        ":cardinality",
        ":compression_ops",
        ":counter",
        ":data_service_ops",
        ":distribute",
        ":distributed_save_op",
        ":enumerate_ops",
        ":error_ops",
        ":from_list",
        ":get_single_element",
        ":global_shuffle_op",
        ":grouping",
        ":interleave_ops",
        ":io",
        ":map_defun",
        ":matching_files",
        ":pad_to_cardinality",
        ":prefetching_ops",
        ":random_access",
        ":readers",
        ":resampling",
        ":scan_ops",
        ":shuffle_ops",
        ":snapshot",
        ":take_while_ops",
        ":unique",
        ":writers",
        "//machina/python/data/ops:dataset_ops",
        "//machina/python/data/ops:debug_mode",
        "//machina/python/data/util:nest",
        "//machina/python/ops:dataset_ops_gen",
    ],
)

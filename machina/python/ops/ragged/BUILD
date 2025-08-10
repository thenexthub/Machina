load("//machina:strict.default.bzl", "py_strict_library", "py_strict_test")
load("//machina:machina.default.bzl", "cuda_py_strict_test")
load("//machina/tools/test:performance.bzl", "tf_py_logged_benchmark")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//machina:internal"],
    licenses = ["notice"],
)

#-------------------------------------------------------------------------------
# RaggedTensor
#-------------------------------------------------------------------------------

py_strict_library(
    name = "ragged",
    srcs = ["__init__.py"],
    tags = ["nofixdeps"],
    deps = [
        ":dynamic_ragged_shape",
        ":ragged_array_ops",
        ":ragged_batch_gather_ops",
        ":ragged_batch_gather_with_default_op",
        ":ragged_check_ops",
        ":ragged_concat_ops",
        ":ragged_config",
        ":ragged_conversion_ops",
        ":ragged_dispatch",
        ":ragged_embedding_ops",
        ":ragged_factory_ops",
        ":ragged_functional_ops",
        ":ragged_gather_ops",
        ":ragged_getitem",
        ":ragged_image_ops",
        ":ragged_map_ops",
        ":ragged_math_ops",
        ":ragged_operators",
        ":ragged_ops",
        ":ragged_string_ops",
        ":ragged_tensor",
        ":ragged_tensor_shape",
        ":ragged_tensor_value",
        ":ragged_util",
        ":ragged_where_op",
        ":segment_id_ops",
    ],
)

py_strict_library(
    name = "ragged_array_ops",
    srcs = ["ragged_array_ops.py"],
    deps = [
        ":dynamic_ragged_shape",
        ":ragged_functional_ops",
        ":ragged_math_ops",
        ":ragged_tensor",
        ":ragged_util",
        ":segment_id_ops",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/framework:sparse_tensor",
        "//machina/python/framework:tensor",
        "//machina/python/framework:tensor_shape",
        "//machina/python/framework:tensor_util",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:array_ops_stack",
        "//machina/python/ops:check_ops",
        "//machina/python/ops:control_flow_ops",
        "//machina/python/ops:data_flow_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:ragged_array_ops_gen",
        "//machina/python/ops:sort_ops",
        "//machina/python/types:core",
        "//machina/python/util:dispatch",
        "//machina/python/util:tf_export",
    ],
)

py_strict_library(
    name = "ragged_autograph",
    srcs = ["ragged_autograph.py"],
    deps = [
        ":ragged_tensor",
        "//machina/python/autograph/operators:control_flow",
        "//machina/python/ops:cond",
    ],
)

py_strict_library(
    name = "ragged_check_ops",
    srcs = ["ragged_check_ops.py"],
    deps = [
        ":ragged_tensor",
        "//machina/python/ops:check_ops",
        "//machina/python/util:dispatch",
    ],
)

py_strict_library(
    name = "ragged_batch_gather_ops",
    srcs = ["ragged_batch_gather_ops.py"],
    deps = [
        ":ragged_gather_ops",
        ":ragged_tensor",
        "//machina/python/ops:array_ops",
        "//machina/python/util:dispatch",
    ],
)

py_strict_library(
    name = "ragged_batch_gather_with_default_op",
    srcs = [
        "ragged_batch_gather_with_default_op.py",
    ],
    deps = [
        ":ragged_array_ops",
        ":ragged_dispatch",
        ":ragged_operators",
        ":ragged_tensor",
        ":ragged_tensor_shape",
        ":ragged_where_op",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:ops",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:check_ops",
        "//machina/python/ops:math_ops",
    ],
)

py_strict_library(
    name = "ragged_bincount_ops",
    srcs = ["ragged_bincount_ops.py"],
    deps = [
        ":ragged_tensor",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/framework:sparse_tensor",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:bincount_ops",
        "//machina/python/ops:check_ops",
        "//machina/python/ops:count_ops_gen",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:math_ops_gen",
        "//machina/python/ops:sparse_ops",
        "//machina/python/util:dispatch",
    ],
)

cuda_py_strict_test(
    name = "ragged_bincount_ops_test",
    size = "small",
    srcs = ["ragged_bincount_ops_test.py"],
    deps = [
        ":ragged_factory_ops",
        ":ragged_tensor",
        "//machina/python/framework:config",
        "//machina/python/framework:errors",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:bincount_ops",
        "//machina/python/ops:sparse_ops",
        "//machina/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_library(
    name = "ragged_concat_ops",
    srcs = ["ragged_concat_ops.py"],
    deps = [
        ":ragged_gather_ops",
        ":ragged_tensor",
        ":ragged_util",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor_shape",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:array_ops_stack",
        "//machina/python/ops:check_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/util:dispatch",
        "//machina/python/util:tf_export",
    ],
)

py_strict_library(
    name = "ragged_conversion_ops",
    srcs = ["ragged_conversion_ops.py"],
    deps = [
        ":ragged_tensor",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:indexed_slices",
        "//machina/python/framework:ops",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:ragged_conversion_ops_gen",
    ],
)

py_strict_library(
    name = "ragged_embedding_ops",
    srcs = ["ragged_embedding_ops.py"],
    deps = [
        ":ragged_array_ops",
        ":ragged_functional_ops",
        ":ragged_tensor",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:array_ops_stack",
        "//machina/python/ops:embedding_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:resource_variable_ops",
        "//machina/python/ops:variables",
        "//machina/python/util:dispatch",
    ],
)

py_strict_library(
    name = "ragged_factory_ops",
    srcs = ["ragged_factory_ops.py"],
    deps = [
        ":ragged_tensor",
        ":ragged_tensor_value",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor_shape",
        "//machina/python/ops:array_ops",
        "//machina/python/util:dispatch",
        "//machina/python/util:numpy_compat",
        "//machina/python/util:tf_export",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "ragged_functional_ops",
    srcs = ["ragged_functional_ops.py"],
    deps = [
        ":ragged_config",
        ":ragged_tensor",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:tensor_shape",
        "//machina/python/util:dispatch",
        "//machina/python/util:tf_export",
    ],
)

py_strict_library(
    name = "ragged_gather_ops",
    srcs = ["ragged_gather_ops.py"],
    deps = [
        ":ragged_array_ops",
        ":ragged_math_ops",
        ":ragged_tensor",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:indexed_slices",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor_shape",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:array_ops_gen",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:ragged_array_ops_gen",
        "//machina/python/util:dispatch",
    ],
)

py_strict_library(
    name = "ragged_getitem",
    srcs = ["ragged_getitem.py"],
    deps = [
        ":ragged_gather_ops",
        ":ragged_math_ops",
        ":ragged_tensor",
        "//machina/python/eager:context",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor",
        "//machina/python/framework:tensor_shape",
        "//machina/python/framework:tensor_util",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:check_ops",
        "//machina/python/ops:cond",
        "//machina/python/ops:math_ops",
        "//machina/python/util:dispatch",
        "//machina/python/util:tf_export",
    ],
)

py_strict_library(
    name = "ragged_math_ops",
    srcs = ["ragged_math_ops.py"],
    deps = [
        ":ragged_functional_ops",
        ":ragged_tensor",
        ":segment_id_ops",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor",
        "//machina/python/framework:tensor_util",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:check_ops",
        "//machina/python/ops:map_fn",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:nn_ops",
        "//machina/python/ops:ragged_math_ops_gen",
        "//machina/python/util:dispatch",
        "//machina/python/util:tf_export",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "ragged_operators",
    srcs = ["ragged_operators.py"],
    deps = [
        ":ragged_getitem",
        ":ragged_tensor",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor",
        "//machina/python/ops:math_ops",
        "//machina/python/util:tf_decorator_py",
    ],
)

py_strict_library(
    name = "ragged_ops",
    srcs = ["ragged_ops.py"],
    deps = [
        ":ragged_array_ops",
        ":ragged_autograph",
        ":ragged_batch_gather_ops",
        ":ragged_batch_gather_with_default_op",
        ":ragged_bincount_ops",
        ":ragged_check_ops",
        ":ragged_concat_ops",
        ":ragged_config",  # fixdeps: keep
        ":ragged_conversion_ops",
        ":ragged_dispatch",
        ":ragged_embedding_ops",
        ":ragged_factory_ops",
        ":ragged_functional_ops",
        ":ragged_gather_ops",
        ":ragged_getitem",
        ":ragged_image_ops",
        ":ragged_map_ops",
        ":ragged_math_ops",
        ":ragged_operators",
        ":ragged_squeeze_op",
        ":ragged_string_ops",
        ":ragged_tensor",
        ":ragged_tensor_shape",
        ":ragged_tensor_value",
        ":ragged_where_op",
        ":segment_id_ops",
    ],
)

py_strict_library(
    name = "ragged_string_ops",
    srcs = ["ragged_string_ops.py"],
    deps = [
        ":ragged_array_ops",
        ":ragged_functional_ops",
        ":ragged_math_ops",
        ":ragged_tensor",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor",
        "//machina/python/framework:tensor_util",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:array_ops_stack",
        "//machina/python/ops:cond",
        "//machina/python/ops:map_fn",
        "//machina/python/ops:string_ops",
        "//machina/python/ops:string_ops_gen",
        "//machina/python/util:compat",
        "//machina/python/util:deprecation",
        "//machina/python/util:dispatch",
        "//machina/python/util:tf_export",
    ],
)

py_strict_library(
    name = "ragged_squeeze_op",
    srcs = ["ragged_squeeze_op.py"],
    deps = [
        ":ragged_tensor",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:control_flow_assert",
        "//machina/python/ops:control_flow_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/util:deprecation",
        "//machina/python/util:dispatch",
    ],
)

py_strict_library(
    name = "ragged_config",
    srcs = ["ragged_config.py"],
)

py_strict_library(
    name = "row_partition",
    srcs = ["row_partition.py"],
    deps = [
        ":segment_id_ops",
        "//machina/core:protos_all_py",
        "//machina/python/framework:composite_tensor",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor",
        "//machina/python/framework:tensor_conversion",
        "//machina/python/framework:tensor_shape",
        "//machina/python/framework:tensor_util",
        "//machina/python/framework:type_spec",
        "//machina/python/framework:type_spec_registry",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:bincount_ops",
        "//machina/python/ops:check_ops",
        "//machina/python/ops:control_flow_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:ragged_math_ops_gen",
        "//machina/python/saved_model:nested_structure_coder",
        "//machina/python/util:tf_export",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "ragged_tensor",
    # RaggedTensor must import ragged_ops to ensure that all dispatched
    # ragged ops are registered.
    # Build cycles from this multi-src build target should not occur
    # if proper TF layering is followed, ie ragged_tensor is allowed to depend on
    # other ops but other ops are not allowed to depend on ragged_tensor.
    srcs = [
        "dynamic_ragged_shape.py",
        "ragged_array_ops.py",
        "ragged_autograph.py",
        "ragged_batch_gather_ops.py",
        "ragged_batch_gather_with_default_op.py",
        "ragged_bincount_ops.py",
        "ragged_check_ops.py",
        "ragged_concat_ops.py",
        "ragged_conversion_ops.py",
        "ragged_dispatch.py",
        "ragged_embedding_ops.py",
        "ragged_factory_ops.py",
        "ragged_functional_ops.py",
        "ragged_gather_ops.py",
        "ragged_getitem.py",
        "ragged_image_ops.py",
        "ragged_map_ops.py",
        "ragged_math_ops.py",
        "ragged_operators.py",
        "ragged_ops.py",
        "ragged_squeeze_op.py",
        "ragged_string_ops.py",
        "ragged_tensor.py",
        "ragged_tensor_shape.py",
        "ragged_where_op.py",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":ragged_config",
        ":ragged_tensor_value",
        ":ragged_util",
        ":row_partition",
        ":segment_id_ops",
        "//machina/core:protos_all_py",
        "//machina/python:tf2",
        "//machina/python/autograph/operators:control_flow",
        "//machina/python/client:session",
        "//machina/python/eager:context",
        "//machina/python/framework:composite_tensor",
        "//machina/python/framework:composite_tensor_gradient",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/framework:extension_type",
        "//machina/python/framework:indexed_slices",
        "//machina/python/framework:ops",
        "//machina/python/framework:sparse_tensor",
        "//machina/python/framework:tensor",
        "//machina/python/framework:tensor_conversion",
        "//machina/python/framework:tensor_shape",
        "//machina/python/framework:tensor_spec",
        "//machina/python/framework:tensor_util",
        "//machina/python/framework:type_spec",
        "//machina/python/framework:type_spec_registry",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:array_ops_gen",
        "//machina/python/ops:array_ops_stack",
        "//machina/python/ops:bincount_ops",
        "//machina/python/ops:check_ops",
        "//machina/python/ops:cond",
        "//machina/python/ops:control_flow_assert",
        "//machina/python/ops:control_flow_ops",
        "//machina/python/ops:count_ops_gen",
        "//machina/python/ops:data_flow_ops",
        "//machina/python/ops:embedding_ops",
        "//machina/python/ops:image_ops",
        "//machina/python/ops:logging_ops",
        "//machina/python/ops:map_fn",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:math_ops_gen",
        "//machina/python/ops:nn_ops",
        "//machina/python/ops:ragged_array_ops_gen",
        "//machina/python/ops:ragged_conversion_ops_gen",
        "//machina/python/ops:ragged_math_ops_gen",
        "//machina/python/ops:sort_ops",
        "//machina/python/ops:sparse_ops",
        "//machina/python/ops:string_ops",
        "//machina/python/ops:string_ops_gen",
        "//machina/python/ops:variables",
        "//machina/python/saved_model:nested_structure_coder",
        "//machina/python/types:core",
        "//machina/python/types:internal",
        "//machina/python/util:compat",
        "//machina/python/util:deprecation",
        "//machina/python/util:dispatch",
        "//machina/python/util:nest",
        "//machina/python/util:numpy_compat",
        "//machina/python/util:tf_decorator_py",
        "//machina/python/util:tf_export",
        "//machina/python/util:tf_inspect",
        "//machina/tools/docs:doc_controls",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "ragged_tensor_shape",
    srcs = ["ragged_tensor_shape.py"],
    deps = [
        ":ragged_array_ops",
        ":ragged_config",
        ":ragged_tensor",
        ":ragged_util",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor_shape",
        "//machina/python/framework:tensor_util",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:array_ops_stack",
        "//machina/python/ops:control_flow_assert",
        "//machina/python/ops:math_ops",
    ],
)

py_strict_library(
    name = "dynamic_ragged_shape",
    srcs = ["dynamic_ragged_shape.py"],
    deps = [
        ":ragged_tensor",
        ":row_partition",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:extension_type",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor",
        "//machina/python/framework:tensor_shape",
        "//machina/python/framework:tensor_util",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:array_ops_stack",
        "//machina/python/ops:check_ops",
        "//machina/python/ops:cond",
        "//machina/python/ops:control_flow_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/types:core",
        "//machina/python/util:dispatch",
        "//machina/python/util:tf_export",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "ragged_tensor_value",
    srcs = ["ragged_tensor_value.py"],
    deps = [
        ":row_partition",
        "//machina/python/util:dispatch",
        "//machina/python/util:tf_export",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "ragged_util",
    srcs = ["ragged_util.py"],
    deps = [
        "//machina/python/ops:array_ops",
        "//machina/python/ops:check_ops",
        "//machina/python/ops:control_flow_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:ragged_math_ops_gen",
    ],
)

py_strict_library(
    name = "ragged_where_op",
    srcs = ["ragged_where_op.py"],
    deps = [
        ":ragged_concat_ops",
        ":ragged_functional_ops",
        ":ragged_gather_ops",
        ":ragged_tensor",
        ":ragged_tensor_shape",
        "//machina/python/framework:ops",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/util:dispatch",
    ],
)

py_strict_library(
    name = "segment_id_ops",
    srcs = ["segment_id_ops.py"],
    deps = [
        ":ragged_util",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor",
        "//machina/python/framework:tensor_shape",
        "//machina/python/framework:tensor_util",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:bincount_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/util:dispatch",
        "//machina/python/util:tf_export",
    ],
)

py_strict_library(
    name = "ragged_image_ops",
    srcs = ["ragged_image_ops.py"],
    deps = [
        ":ragged_tensor",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor_shape",
        "//machina/python/framework:tensor_spec",
        "//machina/python/framework:tensor_util",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:cond",
        "//machina/python/ops:image_ops",
        "//machina/python/ops:map_fn",
        "//machina/python/ops:math_ops",
        "//machina/python/util:dispatch",
    ],
)

py_strict_library(
    name = "ragged_map_ops",
    srcs = ["ragged_map_ops.py"],
    deps = [
        ":ragged_tensor",
        "//machina/python/ops:map_fn",
        "//machina/python/util:nest",
    ],
)

py_strict_library(
    name = "ragged_dispatch",
    srcs = ["ragged_dispatch.py"],
    deps = [
        ":dynamic_ragged_shape",  # fixdeps: keep
        ":ragged_array_ops",  # fixdeps: keep
        ":ragged_batch_gather_ops",  # fixdeps: keep
        ":ragged_concat_ops",  # fixdeps: keep
        ":ragged_gather_ops",  # fixdeps: keep
        ":ragged_math_ops",  # fixdeps: keep
        ":ragged_squeeze_op",  # fixdeps: keep
        ":ragged_tensor",
        ":ragged_tensor_shape",
        ":ragged_util",  # fixdeps: keep
        ":ragged_where_op",  # fixdeps: keep
        "//machina/python/framework:dtypes",  # fixdeps: keep
        "//machina/python/framework:ops",  # fixdeps: keep
        "//machina/python/framework:sparse_tensor",  # fixdeps: keep
        "//machina/python/ops:array_ops",  # fixdeps: keep
        "//machina/python/ops:bitwise_ops",  # fixdeps: keep
        "//machina/python/ops:clip_ops",  # fixdeps: keep
        "//machina/python/ops:logging_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:parsing_ops",  # fixdeps: keep
        "//machina/python/ops:string_ops",
        "//machina/python/ops:variables",  # fixdeps: keep
        "//machina/python/util:dispatch",
        "//machina/python/util:tf_decorator_py",
        "//machina/python/util:tf_export",
        "//machina/python/util:tf_inspect",
        "//third_party/py/numpy",  # fixdeps: keep
    ],
)

py_strict_library(
    name = "ragged_tensor_test_ops",
    srcs = ["ragged_tensor_test_ops.py"],
    deps = [
        "//machina/python/ops:array_ops",
        "//machina/python/ops:bitwise_ops_gen",
        "//machina/python/ops:check_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:nn_impl",
        "//machina/python/ops:nn_ops",
        "//machina/python/ops:parsing_ops",
        "//machina/python/ops:special_math_ops",
        "//machina/python/ops:string_ops",
    ],
)

#-------------------------------------------------------------------------------
# RaggedTensor Tests
#-------------------------------------------------------------------------------

py_strict_test(
    name = "ragged_tensor_test",
    srcs = ["ragged_tensor_test.py"],
    shard_count = 8,
    tags = [
        "no_windows",
    ],
    deps = [
        ":ragged",  # fixdeps: keep
        ":ragged_concat_ops",
        ":ragged_factory_ops",
        ":ragged_gather_ops",
        ":ragged_math_ops",
        ":ragged_tensor",
        ":ragged_tensor_value",
        ":row_partition",
        "//machina/core:protos_all_py",
        "//machina/python/data/ops:dataset_ops",
        "//machina/python/eager:backprop",
        "//machina/python/eager:context",
        "//machina/python/eager:def_function",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor",
        "//machina/python/framework:tensor_shape",
        "//machina/python/framework:test_lib",
        "//machina/python/framework:type_spec",
        "//machina/python/framework:type_utils",
        "//machina/python/ops:array_grad",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:array_ops_stack",
        "//machina/python/ops:cond",
        "//machina/python/ops:gradients_impl",
        "//machina/python/ops:map_fn",
        "//machina/python/ops:math_grad",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:ragged_conversion_ops_gen",
        "//machina/python/ops:tensor_array_grad",
        "//machina/python/platform:test",
        "//machina/python/util:nest",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_test(
    name = "ragged_math_ops_test",
    size = "medium",
    srcs = ["ragged_math_ops_test.py"],
    deps = [
        ":ragged",  # fixdeps: keep
        ":ragged_factory_ops",
        ":ragged_math_ops",
        ":ragged_string_ops",
        ":ragged_tensor",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:nn_ops",
        "//machina/python/platform:test",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_test(
    name = "ragged_getitem_test",
    size = "medium",
    timeout = "long",
    srcs = ["ragged_getitem_test.py"],
    shard_count = 4,
    tags = [
        "no_windows",
    ],
    deps = [
        ":ragged",  # fixdeps: keep
        ":ragged_factory_ops",
        ":ragged_tensor",
        "//machina/python/eager:context",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/platform:test",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_test(
    name = "row_partition_test",
    srcs = ["row_partition_test.py"],
    tags = [
        "no_windows",
    ],
    deps = [
        ":ragged",  # fixdeps: keep
        ":row_partition",
        "//machina/python/eager:context",
        "//machina/python/eager:def_function",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/framework:tensor_shape",
        "//machina/python/framework:tensor_spec",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/platform:test",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_test(
    name = "ragged_eager_test",
    size = "medium",
    srcs = ["ragged_eager_test.py"],
    deps = [
        ":ragged_factory_ops",
        "//machina/python/framework:ops",
        "//machina/python/framework:test_lib",
        "//machina/python/platform:test",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_test(
    name = "ragged_range_op_test",
    srcs = ["ragged_range_op_test.py"],
    deps = [
        ":ragged_math_ops",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:ragged_math_ops_gen",
        "//machina/python/platform:test",
        "//third_party/py/numpy",
    ],
)

py_strict_test(
    name = "ragged_tensor_bounding_shape_op_test",
    srcs = ["ragged_tensor_bounding_shape_op_test.py"],
    deps = [
        ":ragged_factory_ops",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:test_lib",
        "//machina/python/platform:test",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_test(
    name = "ragged_row_lengths_op_test",
    srcs = ["ragged_row_lengths_op_test.py"],
    deps = [
        ":ragged_factory_ops",
        ":ragged_tensor",
        "//machina/python/framework:errors",
        "//machina/python/framework:test_lib",
        "//machina/python/platform:test",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_test(
    name = "ragged_gather_op_test",
    srcs = ["ragged_gather_op_test.py"],
    shard_count = 4,
    deps = [
        ":ragged_factory_ops",
        ":ragged_gather_ops",
        ":ragged_tensor",
        "//machina/python/eager:context",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/framework:indexed_slices",
        "//machina/python/framework:ops",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:gradients_impl",
        "//machina/python/platform:test",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_test(
    name = "ragged_batch_gather_op_test",
    srcs = ["ragged_batch_gather_op_test.py"],
    deps = [
        ":ragged_batch_gather_ops",
        ":ragged_batch_gather_with_default_op",
        ":ragged_factory_ops",
        ":ragged_tensor",
        "//machina/python/eager:context",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/platform:test",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_test(
    name = "ragged_gather_nd_op_test",
    srcs = ["ragged_gather_nd_op_test.py"],
    deps = [
        ":ragged_factory_ops",
        ":ragged_gather_ops",
        "//machina/python/eager:context",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/platform:test",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_test(
    name = "ragged_row_splits_to_segment_ids_op_test",
    srcs = ["ragged_row_splits_to_segment_ids_op_test.py"],
    deps = [
        ":segment_id_ops",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:test_lib",
        "//machina/python/platform:test",
    ],
)

py_strict_test(
    name = "ragged_segment_ids_to_row_splits_op_test",
    srcs = ["ragged_segment_ids_to_row_splits_op_test.py"],
    deps = [
        ":segment_id_ops",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:test_lib",
        "//machina/python/platform:test",
    ],
)

py_strict_test(
    name = "ragged_from_tensor_op_test",
    srcs = ["ragged_from_tensor_op_test.py"],
    deps = [
        ":ragged_tensor",
        "//machina/python/eager:context",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/platform:test",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_test(
    name = "ragged_to_sparse_op_test",
    srcs = ["ragged_to_sparse_op_test.py"],
    tags = [
        "no_windows",
    ],
    deps = [
        ":ragged",  # fixdeps: keep
        ":ragged_factory_ops",
        ":ragged_functional_ops",
        ":ragged_tensor",
        "//machina/python/eager:context",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:gradients_impl",
        "//machina/python/ops:math_ops",
        "//machina/python/platform:test",
    ],
)

py_strict_test(
    name = "ragged_from_sparse_op_test",
    srcs = ["ragged_from_sparse_op_test.py"],
    deps = [
        ":ragged_tensor",
        "//machina/python/eager:context",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/framework:sparse_tensor",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/platform:test",
    ],
)

py_strict_test(
    name = "ragged_to_tensor_op_test",
    srcs = ["ragged_to_tensor_op_test.py"],
    deps = [
        ":ragged_factory_ops",
        ":ragged_tensor",
        "//machina/python/client:session",
        "//machina/python/eager:backprop",
        "//machina/python/eager:context",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/framework:tensor",
        "//machina/python/framework:tensor_shape",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:gradients_impl",
        "//machina/python/platform:benchmark",
        "//machina/python/platform:test",
        "//machina/python/util:nest",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_logged_benchmark(
    name = "ragged_to_tensor_op_benchmark",
    target = "//machina/python/ops/ragged:ragged_to_tensor_op_test",
)

py_strict_test(
    name = "ragged_segment_op_test",
    srcs = ["ragged_segment_op_test.py"],
    deps = [
        ":ragged_factory_ops",
        ":ragged_math_ops",
        ":ragged_tensor",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:errors",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/platform:test",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_test(
    name = "ragged_reduce_op_test",
    srcs = ["ragged_reduce_op_test.py"],
    deps = [
        ":ragged_factory_ops",
        ":ragged_math_ops",
        "//machina/python/eager:context",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/platform:test",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_test(
    name = "ragged_map_flat_values_op_test",
    srcs = ["ragged_map_flat_values_op_test.py"],
    deps = [
        ":ragged_factory_ops",
        ":ragged_functional_ops",
        ":ragged_tensor",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/platform:test",
    ],
)

py_strict_test(
    name = "ragged_const_op_test",
    srcs = ["ragged_const_op_test.py"],
    deps = [
        ":ragged",
        ":ragged_factory_ops",
        ":ragged_tensor",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:test_lib",
        "//machina/python/platform:test",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_test(
    name = "strings_reduce_join_op_test",
    srcs = ["strings_reduce_join_op_test.py"],
    deps = [
        ":ragged_factory_ops",
        ":ragged_string_ops",
        "//machina/python/framework:tensor_shape",
        "//machina/python/framework:test_lib",
        "//machina/python/platform:test",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_test(
    name = "ragged_constant_value_op_test",
    srcs = ["ragged_constant_value_op_test.py"],
    tags = [
        "no_windows",
    ],
    deps = [
        ":ragged_factory_ops",
        ":ragged_tensor_value",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:test_lib",
        "//machina/python/platform:test",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_test(
    name = "convert_to_tensor_or_ragged_tensor_op_test",
    srcs = ["convert_to_tensor_or_ragged_tensor_op_test.py"],
    deps = [
        ":ragged_factory_ops",
        ":ragged_tensor",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:test_lib",
        "//machina/python/platform:test",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_test(
    name = "ragged_boolean_mask_op_test",
    srcs = ["ragged_boolean_mask_op_test.py"],
    deps = [
        ":ragged_array_ops",
        ":ragged_factory_ops",
        "//machina/python/eager:context",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/platform:test",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_test(
    name = "ragged_concat_op_test",
    srcs = ["ragged_concat_op_test.py"],
    deps = [
        ":ragged_concat_ops",
        ":ragged_factory_ops",
        "//machina/python/eager:context",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/platform:test",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_test(
    name = "ragged_stack_op_test",
    srcs = ["ragged_stack_op_test.py"],
    deps = [
        ":ragged_concat_ops",
        ":ragged_factory_ops",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:test_lib",
        "//machina/python/platform:test",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_test(
    name = "ragged_rank_op_test",
    srcs = ["ragged_rank_op_test.py"],
    deps = [
        ":ragged_array_ops",
        ":ragged_factory_ops",
        "//machina/python/framework:test_lib",
        "//machina/python/platform:test",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_test(
    name = "ragged_tile_op_test",
    srcs = ["ragged_tile_op_test.py"],
    deps = [
        ":ragged_array_ops",
        ":ragged_factory_ops",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/platform:test",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_test(
    name = "ragged_util_test",
    srcs = ["ragged_util_test.py"],
    deps = [
        ":ragged_util",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:errors",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/platform:test",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_test(
    name = "ragged_expand_dims_op_test",
    srcs = ["ragged_expand_dims_op_test.py"],
    deps = [
        ":ragged_array_ops",
        ":ragged_factory_ops",
        "//machina/python/framework:test_lib",
        "//machina/python/platform:test",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_test(
    name = "ragged_where_op_test",
    srcs = ["ragged_where_op_test.py"],
    deps = [
        ":ragged_factory_ops",
        ":ragged_where_op",
        "//machina/python/framework:errors",
        "//machina/python/framework:test_lib",
        "//machina/python/platform:test",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_test(
    name = "ragged_dispatch_test",
    srcs = ["ragged_dispatch_test.py"],
    deps = [
        ":dynamic_ragged_shape",
        ":ragged",  # fixdeps: keep
        ":ragged_dispatch",
        ":ragged_factory_ops",
        ":ragged_tensor",
        ":ragged_tensor_test_ops",
        "//machina/python:tf2",
        "//machina/python/eager:context",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/framework:ops",
        "//machina/python/framework:random_seed",
        "//machina/python/framework:sparse_tensor",
        "//machina/python/framework:tensor",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:array_ops_stack",
        "//machina/python/ops:check_ops",
        "//machina/python/ops:clip_ops",
        "//machina/python/ops:data_flow_ops",
        "//machina/python/ops:image_ops_impl",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:nn_impl",
        "//machina/python/ops:nn_ops",
        "//machina/python/ops:string_ops",
        "//machina/python/ops:variables",
        "//machina/python/platform:test",
        "//machina/python/util:dispatch",
        "//machina/python/util:nest",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_test(
    name = "ragged_operators_test",
    srcs = ["ragged_operators_test.py"],
    deps = [
        ":ragged",  # fixdeps: keep
        ":ragged_factory_ops",
        "//machina/python:tf2",
        "//machina/python/framework:ops",
        "//machina/python/framework:test_lib",
        "//machina/python/platform:test",
    ],
)

py_strict_test(
    name = "ragged_resize_image_op_test",
    srcs = ["ragged_resize_image_op_test.py"],
    deps = [
        ":ragged",  # fixdeps: keep
        ":ragged_concat_ops",
        ":ragged_tensor",
        "//machina/python/eager:def_function",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:tensor",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:image_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/platform:test",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_test(
    name = "ragged_map_fn_op_test",
    size = "small",
    srcs = ["ragged_map_fn_op_test.py"],
    deps = [
        ":ragged",  # fixdeps: keep
        ":ragged_factory_ops",
        ":ragged_functional_ops",
        ":ragged_map_ops",
        ":ragged_math_ops",
        ":ragged_tensor",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/framework:sparse_tensor",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:array_ops_stack",
        "//machina/python/ops:map_fn",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:string_ops",
        "//machina/python/platform:test",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_test(
    name = "ragged_tensor_shape_test",
    srcs = ["ragged_tensor_shape_test.py"],
    deps = [
        ":ragged",  # fixdeps: keep
        ":ragged_factory_ops",
        ":ragged_tensor",
        ":ragged_tensor_shape",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:test_lib",
        "//machina/python/platform:test",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_test(
    name = "dynamic_ragged_shape_test",
    size = "medium",
    srcs = ["dynamic_ragged_shape_test.py"],
    shard_count = 8,
    deps = [
        ":dynamic_ragged_shape",
        ":ragged",  # fixdeps: keep
        ":ragged_array_ops",
        ":ragged_factory_ops",
        ":ragged_tensor",
        ":row_partition",
        "//machina/python/client:session",
        "//machina/python/data/ops:dataset_ops",
        "//machina/python/eager:context",
        "//machina/python/eager:def_function",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor",
        "//machina/python/framework:tensor_shape",
        "//machina/python/framework:tensor_spec",
        "//machina/python/framework:tensor_util",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:array_ops_stack",
        "//machina/python/ops:gradients_impl",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:math_ops_gen",
        "//machina/python/ops:string_ops",
        "//machina/python/platform:test",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_test(
    name = "ragged_size_op_test",
    srcs = ["ragged_size_op_test.py"],
    deps = [
        ":ragged_array_ops",
        ":ragged_factory_ops",
        "//machina/python/framework:test_lib",
        "//machina/python/platform:test",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_test(
    name = "ragged_placeholder_op_test",
    srcs = ["ragged_placeholder_op_test.py"],
    deps = [
        ":ragged_factory_ops",
        "//machina/python/eager:context",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/framework:test_lib",
        "//machina/python/platform:test",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_test(
    name = "ragged_squeeze_op_test",
    srcs = ["ragged_squeeze_op_test.py"],
    deps = [
        ":ragged_conversion_ops",
        ":ragged_factory_ops",
        ":ragged_squeeze_op",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:errors",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/platform:test",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_strict_test(
    name = "ragged_dynamic_partition_op_test",
    srcs = ["ragged_dynamic_partition_op_test.py"],
    deps = [
        ":ragged_array_ops",
        ":ragged_concat_ops",
        ":ragged_factory_ops",
        "//machina/python/eager:context",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:data_flow_ops",
        "//machina/python/platform:test",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_test(
    name = "ragged_merge_dims_op_test",
    srcs = ["ragged_merge_dims_op_test.py"],
    deps = [
        ":ragged_factory_ops",
        "//machina/python/eager:context",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/platform:test",
        "//machina/python/util:nest",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_test(
    name = "string_ngrams_op_test",
    size = "small",
    srcs = ["string_ngrams_op_test.py"],
    deps = [
        ":ragged_factory_ops",
        ":ragged_string_ops",
        "//machina/python/eager:def_function",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:tensor",
        "//machina/python/framework:test_lib",
        "//machina/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_test(
    name = "ragged_reverse_op_test",
    srcs = ["ragged_reverse_op_test.py"],
    deps = [
        ":ragged_array_ops",
        ":ragged_factory_ops",
        "//machina/python/framework:test_lib",
        "//machina/python/platform:test",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_test(
    name = "ragged_cross_op_test",
    srcs = ["ragged_cross_op_test.py"],
    tags = [
        "no_windows",  # TODO(b/150702952): Reenable windows test once fixed.
    ],
    deps = [
        ":ragged_array_ops",
        ":ragged_factory_ops",
        ":ragged_tensor",
        "//machina/python/eager:def_function",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/framework:ops",
        "//machina/python/framework:sparse_tensor",
        "//machina/python/framework:tensor_spec",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:ragged_array_ops_gen",
        "//machina/python/ops:random_ops",
        "//machina/python/ops:sparse_ops",
        "//machina/python/platform:test",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_test(
    name = "ragged_one_hot_op_test",
    srcs = ["ragged_one_hot_op_test.py"],
    deps = [
        ":ragged_array_ops",
        ":ragged_factory_ops",
        ":ragged_tensor",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/framework:tensor_shape",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/platform:test",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_test(
    name = "ragged_print_op_test",
    srcs = ["ragged_print_op_test.py"],
    deps = [
        ":ragged",  # fixdeps: keep
        ":ragged_factory_ops",
        ":ragged_string_ops",
        ":ragged_tensor",
        "//machina/python/eager:def_function",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:logging_ops",
        "//machina/python/ops:sparse_ops",
        "//machina/python/platform:test",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_test(
    name = "ragged_tensor_supported_values_test",
    srcs = ["ragged_tensor_supported_values_test.py"],
    shard_count = 10,
    deps = [
        ":ragged_factory_ops",
        ":ragged_tensor",
        ":ragged_tensor_test_ops",
        "//machina/python/eager:context",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:extension_type",
        "//machina/python/framework:tensor",
        "//machina/python/framework:tensor_shape",
        "//machina/python/framework:test_lib",
        "//machina/python/framework:type_spec",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:check_ops",
        "//machina/python/ops:clip_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:nn_ops",
        "//machina/python/ops:string_ops",
        "//machina/python/platform:test",
        "//machina/python/util:dispatch",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_test(
    name = "ragged_matmul_op_test",
    srcs = ["ragged_matmul_op_test.py"],
    deps = [
        ":ragged_concat_ops",
        ":ragged_factory_ops",
        ":ragged_math_ops",
        ":ragged_tensor",
        "//machina/python/eager:context",
        "//machina/python/eager:def_function",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/platform:test",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_test(
    name = "ragged_split_op_test",
    srcs = ["ragged_split_op_test.py"],
    deps = [
        ":ragged_array_ops",
        ":ragged_tensor",
        "//machina/python/eager:def_function",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor_spec",
        "//machina/python/framework:test_lib",
        "//machina/python/platform:test",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_strict_test(
    name = "ragged_factory_ops_test",
    srcs = ["ragged_factory_ops_test.py"],
    deps = [
        ":ragged_factory_ops",
        "//machina/python/data/ops:dataset_ops",
        "//machina/python/data/ops:multi_device_iterator_ops",
        "//machina/python/distribute:mirrored_strategy",
        "//machina/python/eager:def_function",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:map_fn",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:string_ops",
        "//machina/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_test(
    name = "ragged_bitcast_op_test",
    srcs = ["ragged_bitcast_op_test.py"],
    deps = [
        ":ragged_array_ops",
        ":ragged_factory_ops",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:test_lib",
        "//machina/python/platform:test",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_test(
    name = "ragged_fill_empty_rows_op_test",
    srcs = ["ragged_fill_empty_rows_op_test.py"],
    deps = [
        ":ragged_array_ops",
        ":ragged_factory_ops",
        ":ragged_tensor",
        "//machina/python/eager:def_function",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:gradient_checker",
        "//machina/python/platform:test",
        "//third_party/py/numpy",
    ],
)

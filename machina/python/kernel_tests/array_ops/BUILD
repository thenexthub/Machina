# Tests of TensorFlow array ops kernels written using the Python API.

load("//machina:machina.default.bzl", "cuda_py_strict_test", "tf_py_strict_test")
load("//machina/core/platform:build_config_root.bzl", "tf_additional_xla_deps_py")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    licenses = ["notice"],
)

cuda_py_strict_test(
    name = "array_ops_test",
    size = "medium",
    srcs = ["array_ops_test.py"],
    shard_count = 10,
    tags = [
        "optonly",  # times out
    ],
    deps = [
        "//machina/python/client:session",
        "//machina/python/eager:backprop",
        "//machina/python/eager:context",
        "//machina/python/eager:def_function",
        "//machina/python/framework:config",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/framework:ops",
        "//machina/python/framework:sparse_tensor",
        "//machina/python/framework:tensor",
        "//machina/python/framework:tensor_shape",
        "//machina/python/framework:test_lib",
        "//machina/python/framework:test_ops",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:array_ops_gen",
        "//machina/python/ops:array_ops_stack",
        "//machina/python/ops:gradient_checker_v2",
        "//machina/python/ops:init_ops",
        "//machina/python/ops:list_ops",
        "//machina/python/ops:map_fn",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:random_ops",
        "//machina/python/ops:resource_variable_ops",
        "//machina/python/ops:state_ops",
        "//machina/python/ops:variable_scope",
        "//machina/python/ops:variable_v1",
        "//machina/python/ops:variables",
        "//machina/python/ops/ragged:ragged_tensor",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_strict_test(
    name = "batch_gather_op_test",
    srcs = ["batch_gather_op_test.py"],
    deps = [
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
    name = "batch_scatter_ops_test",
    srcs = ["batch_scatter_ops_test.py"],
    deps = [
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:indexed_slices",
        "//machina/python/ops:state_ops",
        "//machina/python/ops:variables",
        "//machina/python/platform:client_testlib",
    ],
)

cuda_py_strict_test(
    name = "batchtospace_op_test",
    size = "small",
    srcs = ["batchtospace_op_test.py"],
    deps = [
        "//machina/python/framework:constant_op",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:array_ops_gen",
        "//machina/python/ops:gradient_checker",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
    name = "bcast_ops_test",
    size = "small",
    srcs = ["bcast_ops_test.py"],
    deps = [
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/ops:array_ops_gen",
        "//machina/python/platform:client_testlib",
    ],
)

cuda_py_strict_test(
    name = "bitcast_op_test",
    size = "small",
    srcs = ["bitcast_op_test.py"],
    deps = [
        "//machina/python/framework:errors",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "broadcast_to_ops_test",
    size = "medium",
    srcs = ["broadcast_to_ops_test.py"],
    deps = [
        "//machina/python/eager:context",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:gradient_checker_v2",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "cast_op_test",
    size = "small",
    srcs = ["cast_op_test.py"],
    xla_tags = [
        "no_cuda_asan",  # times out
    ],
    deps = [
        "//machina/python/eager:context",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:sparse_tensor",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:gradient_checker_v2",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:state_ops_gen",
        "//machina/python/ops:variables",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "concat_op_test",
    size = "medium",
    srcs = ["concat_op_test.py"],
    shard_count = 4,
    tags = ["no_windows"],  # b/126916429
    xla_tags = [
        "no_cuda_asan",  # times out
    ],
    deps = [
        "//machina/python/eager:def_function",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:errors",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:array_ops_gen",
        "//machina/python/ops:gradient_checker",
        "//machina/python/ops:gradients_impl",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:nn_ops",
        "//machina/python/ops:variables",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "constant_op_eager_test",
    size = "small",
    srcs = ["constant_op_eager_test.py"],
    tags = [
        "no_windows",  # TODO(b/207048720): re-enable
    ],
    deps = [
        "//machina/python/eager:context",
        "//machina/python/eager:test",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:errors",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/util:compat",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "constant_op_test",
    size = "small",
    srcs = ["constant_op_test.py"],
    deps = [
        "//machina/core:protos_all_py",
        "//machina/python/eager:def_function",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/framework:importer",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor",
        "//machina/python/framework:tensor_shape",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:gradient_checker",
        "//machina/python/ops:logging_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/platform:client_testlib",
        "//machina/python/util:compat",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "denormal_test",
    size = "small",
    srcs = ["denormal_test.py"],
    deps = [
        "//machina/python/framework:constant_op",
        "//machina/python/framework:test_lib",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "depthtospace_op_test",
    size = "medium",
    srcs = ["depthtospace_op_test.py"],
    tags = [
        "no_windows",  # TODO(b/184424727): Re-enable this.
        "no_windows_gpu",
    ],
    xla_tags = [
        "no_cuda_asan",  # times out
    ],
    deps = [
        "//machina/python/client:device_lib",
        "//machina/python/eager:context",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:errors",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:array_ops_gen",
        "//machina/python/ops:gradient_checker",
        "//machina/python/ops:math_ops",
        "//machina/python/platform:client_testlib",
        "//machina/python/platform:tf_logging",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_strict_test(
    name = "diag_op_test",
    size = "medium",
    srcs = ["diag_op_test.py"],
    shard_count = 6,
    tags = ["no_windows_gpu"],
    deps = [
        "//machina/python/framework:constant_op",
        "//machina/python/framework:errors",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:gradient_checker",
        "//machina/python/ops:gradients_impl",
        "//machina/python/platform:client_testlib",
        "//machina/python/platform:tf_logging",
        "//third_party/py/numpy",
    ],
)

tf_py_strict_test(
    name = "edit_distance_op_test",
    size = "small",
    srcs = ["edit_distance_op_test.py"],
    deps = [
        "//machina/python/eager:def_function",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:errors",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:sparse_tensor",
        "//machina/python/ops:array_ops",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

tf_py_strict_test(
    name = "fingerprint_op_test",
    size = "small",
    srcs = ["fingerprint_op_test.py"],
    deps = [
        "//machina/python/ops:array_ops",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "gather_nd_op_test",
    size = "small",
    srcs = ["gather_nd_op_test.py"],
    deps = [
        "//machina/python/client:session",
        "//machina/python/eager:context",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:indexed_slices",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:gradients_impl",
        "//machina/python/ops:ref_variable",
        "//machina/python/ops:resource_variable_ops",
        "//machina/python/ops:variables",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "gather_op_test",
    size = "medium",
    srcs = ["gather_op_test.py"],
    shard_count = 3,
    deps = [
        "//machina/python/eager:backprop",
        "//machina/python/eager:def_function",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:errors",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:indexed_slices",
        "//machina/python/framework:tensor_spec",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:gradient_checker_v2",
        "//machina/python/ops:gradients_impl",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:ref_variable",
        "//machina/python/ops:resource_variable_ops",
        "//machina/python/ops:variables",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ] + tf_additional_xla_deps_py(),
)

cuda_py_strict_test(
    name = "huge_slice_op_test",
    size = "medium",
    srcs = ["huge_slice_op_test.py"],
    tags = [
        "no_oss",  # Requires 4GB+ RAM
    ],
    deps = [
        "//machina/python/framework:constant_op",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/ops:array_ops",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

tf_py_strict_test(
    name = "identity_n_op_py_test",
    size = "small",
    srcs = ["identity_n_op_py_test.py"],
    deps = [
        "//machina/python/framework:constant_op",
        "//machina/python/ops:array_ops",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

tf_py_strict_test(
    name = "identity_op_py_test",
    size = "small",
    srcs = ["identity_op_py_test.py"],
    deps = [
        "//machina/python/framework:constant_op",
        "//machina/python/framework:sparse_tensor",
        "//machina/python/ops:array_ops",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "init_ops_test",
    size = "medium",
    srcs = ["init_ops_test.py"],
    shard_count = 4,
    deps = [
        "//machina/python/framework:constant_op",
        "//machina/python/framework:errors",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:random_seed",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:init_ops",
        "//machina/python/ops:linalg_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:partitioned_variables",
        "//machina/python/ops:random_ops",
        "//machina/python/ops:variable_scope",
        "//machina/python/ops:variables",
        "//machina/python/platform:client_testlib",
        "//machina/python/util:numpy_compat",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "inplace_ops_test",
    size = "small",
    srcs = ["inplace_ops_test.py"],
    shard_count = 10,
    deps = [
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/framework:ops",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:inplace_ops",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "large_concat_op_test",
    size = "medium",
    srcs = ["large_concat_op_test.py"],
    deps = [
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/ops:array_ops",
        "//machina/python/platform:client_testlib",
    ],
)

cuda_py_strict_test(
    name = "manip_ops_test",
    size = "medium",
    srcs = ["manip_ops_test.py"],
    tags = [
        "no_windows_gpu",
    ],
    deps = [
        "//machina/python/framework:constant_op",
        "//machina/python/framework:errors",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:gradient_checker",
        "//machina/python/ops:manip_ops",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
        "@pypi_packaging//:pkg",
    ],
)

cuda_py_strict_test(
    name = "matrix_band_part_op_test",
    size = "medium",
    srcs = ["matrix_band_part_op_test.py"],
    shard_count = 20,
    xla_tags = [
        "no_cuda_asan",  # times out
    ],
    deps = [
        "//machina/python/client:session",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:control_flow_ops",
        "//machina/python/ops:gradient_checker",
        "//machina/python/ops:variables",
        "//machina/python/platform:benchmark",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "one_hot_op_test",
    size = "small",
    srcs = ["one_hot_op_test.py"],
    tags = ["no_windows_gpu"],
    xla_tags = [
        "no_cuda_asan",  # times out
    ],
    deps = [
        "//machina/python/framework:constant_op",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/ops:array_ops",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_strict_test(
    name = "pad_op_test",
    size = "medium",
    timeout = "moderate",
    srcs = ["pad_op_test.py"],
    tags = ["no_mac"],  # test is times out on mac b/186262388
    xla_tags = [
        "no_cuda_asan",  # times out
    ],
    deps = [
        "//machina/python/framework:constant_op",
        "//machina/python/framework:errors",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:gradient_checker_v2",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "reshape_op_test",
    size = "small",
    srcs = ["reshape_op_test.py"],
    deps = [
        "//machina/python/framework:constant_op",
        "//machina/python/framework:errors",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:gradient_checker_v2",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "reverse_sequence_op_test",
    size = "small",
    srcs = ["reverse_sequence_op_test.py"],
    deps = [
        "//machina/python/eager:context",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:errors",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:gradient_checker_v2",
        "//machina/python/platform:client_testlib",
        "//machina/python/util:nest",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "scalar_test",
    size = "small",
    srcs = ["scalar_test.py"],
    # b/140221961: Invalid dims for operations
    xla_enable_strict_auto_jit = False,
    deps = [
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:io_ops_gen",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:nn_grad",
        "//machina/python/ops:random_ops",
        "//machina/python/ops:sparse_ops",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "scatter_nd_ops_test",
    size = "medium",
    srcs = ["scatter_nd_ops_test.py"],
    shard_count = 2,
    deps = [
        "//machina/python/eager:context",
        "//machina/python/eager:def_function",
        "//machina/python/framework:config",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:errors",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:gradient_checker_v2",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:resource_variable_ops",
        "//machina/python/ops:state_ops",
        "//machina/python/ops:variable_v1",
        "//machina/python/ops:variables",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_strict_test(
    name = "scatter_ops_test",
    size = "medium",  # NOTE: This is not run by default.
    srcs = ["scatter_ops_test.py"],
    shard_count = 2,
    tags = [
        "no_oss",  # TODO(b/282985589)
        "noasan",
        "notsan",
        "optonly",
    ],
    deps = [
        "//machina/python/framework:constant_op",
        "//machina/python/framework:errors",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:ref_variable",
        "//machina/python/ops:state_ops",
        "//machina/python/ops:variables",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "shape_ops_test",
    size = "medium",
    srcs = ["shape_ops_test.py"],
    deps = [
        "//machina/core:protos_all_py",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/framework:importer",
        "//machina/python/framework:sparse_tensor",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:gradient_checker",
        "//machina/python/ops:gradients_impl",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_strict_test(
    name = "slice_op_test",
    size = "medium",
    srcs = ["slice_op_test.py"],
    deps = [
        "//machina/python/eager:backprop",
        "//machina/python/eager:def_function",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:errors",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:gradients_impl",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:nn_ops",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "spacetobatch_op_test",
    size = "small",
    srcs = ["spacetobatch_op_test.py"],
    xla_tags = [
        "no_cuda_asan",  # times out
    ],
    deps = [
        "//machina/python/framework:constant_op",
        "//machina/python/framework:errors",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:tensor_util",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:array_ops_gen",
        "//machina/python/ops:gradient_checker",
        "//machina/python/ops:math_ops",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_strict_test(
    name = "spacetodepth_op_test",
    size = "medium",
    srcs = ["spacetodepth_op_test.py"],
    tags = [
        "no_windows",
        "no_windows_gpu",
    ],
    deps = [
        "//machina/python/framework:constant_op",
        "//machina/python/framework:errors",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:array_ops_gen",
        "//machina/python/ops:gradient_checker_v2",
        "//machina/python/ops:math_ops",
        "//machina/python/platform:client_testlib",
        "//machina/python/platform:tf_logging",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "split_op_test",
    size = "medium",
    srcs = ["split_op_test.py"],
    shard_count = 4,
    deps = [
        "//machina/python/framework:constant_op",
        "//machina/python/framework:errors",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:gradients_impl",
        "//machina/python/ops:math_ops",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "stack_op_test",
    size = "medium",
    srcs = ["stack_op_test.py"],
    deps = [
        "//machina/python:tf2",
        "//machina/python/eager:context",
        "//machina/python/eager:def_function",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:errors",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:array_ops_gen",
        "//machina/python/ops:array_ops_stack",
        "//machina/python/ops:gradient_checker_v2",
        "//machina/python/ops:math_ops",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "unique_op_test",
    size = "small",
    srcs = ["unique_op_test.py"],
    tags = [
        "no_windows",  # TODO(b/192259045)
    ],
    deps = [
        "//machina/python/framework:dtypes",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:array_ops_gen",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "unstack_op_test",
    size = "small",
    srcs = ["unstack_op_test.py"],
    tags = [
        "no_cuda_asan",  # b/173806679
    ],
    xla_tags = [
        "no_cuda_asan",  # times out
    ],
    deps = [
        "//machina/python/framework:constant_op",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:array_ops_stack",
        "//machina/python/ops:gradient_checker_v2",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

tf_py_strict_test(
    name = "weights_broadcast_test",
    size = "small",
    srcs = ["weights_broadcast_test.py"],
    shard_count = 3,
    deps = [
        "//machina/python/framework:errors",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:weights_broadcast_ops",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "where_op_test",
    size = "medium",
    srcs = ["where_op_test.py"],
    tags = ["no_cuda_asan"],  #TODO(b/212580469)
    deps = [
        "//machina/python/client:session",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:random_ops",
        "//machina/python/ops:resource_variable_ops",
        "//machina/python/platform:benchmark",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

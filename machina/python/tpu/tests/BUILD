# Description: Tests defined for Cloud TPUs

load("//machina:pytype.default.bzl", "pytype_strict_library")
load("//machina/python/tpu:tpu.bzl", "tpu_py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    licenses = ["notice"],
)

pytype_strict_library(
    name = "tpu_embedding_base_test",
    srcs = ["tpu_embedding_base_test.py"],
    deps = [
        "//machina/python/data/ops:dataset_ops",
        "//machina/python/distribute:tpu_strategy",
        "//machina/python/distribute/cluster_resolver:tpu_cluster_resolver_py",
        "//machina/python/eager:remote",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:sparse_tensor",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:init_ops_v2",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:math_ops_gen",
        "//machina/python/ops/ragged:ragged_tensor",
        "//machina/python/platform:client_testlib",
        "//machina/python/tpu:tpu_embedding_v2",
        "//machina/python/tpu:tpu_embedding_v2_utils",
        "//machina/python/util:nest",
        "//third_party/py/numpy",
        "@absl_py//absl/flags",
        "@absl_py//absl/testing:parameterized",
    ],
)

tpu_py_strict_test(
    name = "tpu_embedding_v2_checkpoint_test",
    srcs = [
        "tpu_embedding_v2_checkpoint_test.py",
    ],
    disable_experimental = True,
    disable_mlir_bridge = False,
    deps = [
        ":tpu_embedding_base_test",
        "//machina/python/checkpoint",
        "//machina/python/compat:v2_compat",
        "//machina/python/distribute/cluster_resolver:tpu_cluster_resolver_py",
        "//machina/python/eager:def_function",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor_spec",
        "//machina/python/module",
        "//machina/python/ops:init_ops_v2",
        "//machina/python/platform:client_testlib",
        "//machina/python/saved_model:load",
        "//machina/python/saved_model:save",
        "//machina/python/tpu:tpu_embedding_for_serving",
        "//machina/python/tpu:tpu_embedding_v2",
        "//machina/python/tpu:tpu_embedding_v2_utils",
        "//machina/python/training:checkpoint_utils",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tpu_py_strict_test(
    name = "tpu_embedding_v2_mp_strategy_test",
    srcs = [
        "tpu_embedding_v2_mp_strategy_test.py",
    ],
    disable_experimental = True,
    disable_mlir_bridge = False,
    deps = [
        ":tpu_embedding_base_test",
        "//machina/python/compat:v2_compat",
        "//machina/python/data/ops:dataset_ops",
        "//machina/python/distribute:distribute_lib",
        "//machina/python/distribute:tpu_strategy",
        "//machina/python/eager:backprop",
        "//machina/python/eager:def_function",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/ops:init_ops_v2",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:math_ops_gen",
        "//machina/python/ops:variables",
        "//machina/python/platform:client_testlib",
        "//machina/python/tpu:device_assignment",
        "//machina/python/tpu:tpu_embedding_v2",
        "//machina/python/tpu:tpu_embedding_v2_utils",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tpu_py_strict_test(
    name = "tpu_embedding_v2_enqueue_mode_test",
    srcs = [
        "tpu_embedding_v2_enqueue_mode_test.py",
    ],
    disable_experimental = True,
    disable_mlir_bridge = False,
    deps = [
        ":tpu_embedding_base_test",
        "//machina/python/compat:v2_compat",
        "//machina/python/distribute:distribute_lib",
        "//machina/python/eager:def_function",
        "//machina/python/framework:config",
        "//machina/python/framework:tensor_shape",
        "//machina/python/ops:array_ops",
        "//machina/python/platform:client_testlib",
        "//machina/python/util:nest",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tpu_py_strict_test(
    name = "tpu_embedding_v2_invalid_input_test",
    srcs = [
        "tpu_embedding_v2_invalid_input_test.py",
    ],
    disable_experimental = True,
    disable_mlir_bridge = False,
    deps = [
        ":tpu_embedding_base_test",
        "//machina/python/compat:v2_compat",
        "//machina/python/distribute:distribute_lib",
        "//machina/python/eager:def_function",
        "//machina/python/framework:config",
        "//machina/python/platform:client_testlib",
        "//machina/python/tpu:tpu_embedding_v2",
        "//machina/python/tpu:tpu_embedding_v2_utils",
        "@absl_py//absl/testing:parameterized",
    ],
)

tpu_py_strict_test(
    name = "tpu_embedding_v2_valid_input_test",
    srcs = [
        "tpu_embedding_v2_valid_input_test.py",
    ],
    disable_experimental = True,
    disable_mlir_bridge = False,
    deps = [
        ":tpu_embedding_base_test",
        "//machina/python/compat:v2_compat",
        "//machina/python/data/ops:dataset_ops",
        "//machina/python/distribute:distribute_lib",
        "//machina/python/eager:def_function",
        "//machina/python/framework:sparse_tensor",
        "//machina/python/framework:tensor_shape",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:init_ops_v2",
        "//machina/python/ops/ragged:ragged_tensor",
        "//machina/python/platform:client_testlib",
        "//machina/python/tpu:tpu_embedding_v2",
        "//machina/python/tpu:tpu_embedding_v2_utils",
        "//machina/python/util:nest",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tpu_py_strict_test(
    name = "tpu_embedding_v2_hd_valid_input_test",
    srcs = [
        "tpu_embedding_v2_hd_valid_input_test.py",
    ],
    disable_experimental = True,
    disable_mlir_bridge = False,
    deps = [
        ":tpu_embedding_base_test",
        "//machina/python/compat:v2_compat",
        "//machina/python/data/ops:dataset_ops",
        "//machina/python/distribute:distribute_lib",
        "//machina/python/eager:def_function",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:tensor_shape",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

tpu_py_strict_test(
    name = "tpu_embedding_v2_hd_invalid_input_test",
    srcs = [
        "tpu_embedding_v2_hd_invalid_input_test.py",
    ],
    disable_experimental = True,
    disable_mlir_bridge = False,
    deps = [
        ":tpu_embedding_base_test",
        "//machina/python/compat:v2_compat",
        "//machina/python/distribute:distribute_lib",
        "//machina/python/eager:def_function",
        "//machina/python/framework:tensor_shape",
        "//machina/python/platform:client_testlib",
    ],
)

tpu_py_strict_test(
    name = "tpu_embedding_v2_sequence_feature_test",
    srcs = [
        "tpu_embedding_v2_sequence_feature_test.py",
    ],
    disable_experimental = True,
    disable_mlir_bridge = False,
    deps = [
        ":tpu_embedding_base_test",
        "//machina/python/compat:v2_compat",
        "//machina/python/distribute:distribute_lib",
        "//machina/python/eager:def_function",
        "//machina/python/framework:tensor_shape",
        "//machina/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

pytype_strict_library(
    name = "tpu_embedding_v2_correctness_base_test",
    srcs = ["tpu_embedding_v2_correctness_base_test.py"],
    deps = [
        ":tpu_embedding_base_test",
        "//machina/python/compat:v2_compat",
        "//machina/python/distribute:distribute_lib",
        "//machina/python/eager:backprop",
        "//machina/python/eager:def_function",
        "//machina/python/framework:tensor_shape",
        "//machina/python/platform:client_testlib",
    ],
)

tpu_py_strict_test(
    name = "tpu_embedding_v2_correctness_sparse_training_test",
    srcs = [
        "tpu_embedding_v2_correctness_sparse_training_test.py",
    ],
    disable_experimental = True,
    disable_mlir_bridge = False,
    deps = [
        ":tpu_embedding_v2_correctness_base_test",
        "//machina/python/compat:v2_compat",
        "//machina/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

tpu_py_strict_test(
    name = "tpu_embedding_v2_correctness_sparse_forward_test",
    srcs = [
        "tpu_embedding_v2_correctness_sparse_forward_test.py",
    ],
    disable_experimental = True,
    disable_mlir_bridge = False,
    deps = [
        ":tpu_embedding_v2_correctness_base_test",
        "//machina/python/compat:v2_compat",
        "//machina/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

tpu_py_strict_test(
    name = "tpu_embedding_v2_correctness_ragged_training_test",
    srcs = [
        "tpu_embedding_v2_correctness_ragged_training_test.py",
    ],
    disable_experimental = True,
    disable_mlir_bridge = False,
    deps = [
        ":tpu_embedding_v2_correctness_base_test",
        "//machina/python/compat:v2_compat",
        "//machina/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

tpu_py_strict_test(
    name = "tpu_embedding_v2_correctness_ragged_forward_test",
    srcs = [
        "tpu_embedding_v2_correctness_ragged_forward_test.py",
    ],
    disable_experimental = True,
    disable_mlir_bridge = False,
    deps = [
        ":tpu_embedding_v2_correctness_base_test",
        "//machina/python/compat:v2_compat",
        "//machina/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

tpu_py_strict_test(
    name = "tpu_embedding_v2_correctness_hd_sparse_training_test",
    srcs = [
        "tpu_embedding_v2_correctness_hd_sparse_training_test.py",
    ],
    disable_experimental = True,
    disable_mlir_bridge = False,
    deps = [
        ":tpu_embedding_v2_correctness_base_test",
        "//machina/python/compat:v2_compat",
        "//machina/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

tpu_py_strict_test(
    name = "tpu_embedding_v2_correctness_hd_sparse_forward_test",
    srcs = [
        "tpu_embedding_v2_correctness_hd_sparse_forward_test.py",
    ],
    disable_experimental = True,
    disable_mlir_bridge = False,
    deps = [
        ":tpu_embedding_v2_correctness_base_test",
        "//machina/python/compat:v2_compat",
        "//machina/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

tpu_py_strict_test(
    name = "tpu_embedding_v2_correctness_hd_ragged_training_test",
    srcs = [
        "tpu_embedding_v2_correctness_hd_ragged_training_test.py",
    ],
    disable_experimental = True,
    disable_mlir_bridge = False,
    deps = [
        ":tpu_embedding_v2_correctness_base_test",
        "//machina/python/compat:v2_compat",
        "//machina/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

tpu_py_strict_test(
    name = "tpu_embedding_v2_correctness_hd_ragged_forward_test",
    srcs = [
        "tpu_embedding_v2_correctness_hd_ragged_forward_test.py",
    ],
    disable_experimental = True,
    disable_mlir_bridge = False,
    deps = [
        ":tpu_embedding_v2_correctness_base_test",
        "//machina/python/compat:v2_compat",
        "//machina/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

tpu_py_strict_test(
    name = "tpu_embedding_v2_correctness_dense_lookup_test",
    srcs = [
        "tpu_embedding_v2_correctness_dense_lookup_test.py",
    ],
    disable_experimental = True,
    disable_mlir_bridge = False,
    deps = [
        ":tpu_embedding_v2_correctness_base_test",
        "//machina/python/compat:v2_compat",
        "//machina/python/distribute:distribute_lib",
        "//machina/python/eager:def_function",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tpu_py_strict_test(
    name = "tpu_embedding_v2_correctness_sequence_feature_test",
    srcs = [
        "tpu_embedding_v2_correctness_sequence_feature_test.py",
    ],
    disable_experimental = True,
    disable_mlir_bridge = False,
    deps = [
        ":tpu_embedding_v2_correctness_base_test",
        "//machina/python/compat:v2_compat",
        "//machina/python/distribute:distribute_lib",
        "//machina/python/eager:def_function",
        "//machina/python/ops:array_ops",
        "//machina/python/platform:client_testlib",
        "//machina/python/tpu:tpu_embedding_v2",
        "//machina/python/tpu:tpu_embedding_v2_utils",
        "//machina/python/util:nest",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tpu_py_strict_test(
    name = "tpu_embedding_v2_initialization_test",
    srcs = [
        "tpu_embedding_v2_initialization_test.py",
    ],
    disable_experimental = True,
    disable_mlir_bridge = False,
    deps = [
        ":tpu_embedding_base_test",
        "//machina/python/compat:v2_compat",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:sparse_tensor",
        "//machina/python/framework:tensor_shape",
        "//machina/python/ops:init_ops_v2",
        "//machina/python/platform:client_testlib",
        "//machina/python/tpu:tpu_embedding_v2",
        "//machina/python/tpu:tpu_embedding_v2_utils",
        "//third_party/py/numpy",
    ],
)

### tpu embedding v1 tests

tpu_py_strict_test(
    name = "tpu_embedding_v1_checkpoint_test",
    srcs = [
        "tpu_embedding_v1_checkpoint_test.py",
    ],
    disable_mlir_bridge = False,
    tags = ["no_oss"],
    deps = [
        ":tpu_embedding_base_test",
        "//machina/python/checkpoint",
        "//machina/python/compat:v2_compat",
        "//machina/python/eager:def_function",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:tensor_spec",
        "//machina/python/ops:init_ops_v2",
        "//machina/python/platform:client_testlib",
        "//machina/python/saved_model:load",
        "//machina/python/saved_model:save",
        "//machina/python/tpu:tpu_embedding_for_serving",
        "//machina/python/tpu:tpu_embedding_v1",
        "//machina/python/tpu:tpu_embedding_v2_utils",
        "//machina/python/training:checkpoint_utils",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tpu_py_strict_test(
    name = "tpu_embedding_v1_correctness_test",
    srcs = [
        "tpu_embedding_v1_correctness_test.py",
    ],
    disable_mlir_bridge = False,
    tags = ["no_oss"],
    deps = [
        ":tpu_embedding_base_test",
        "//machina/python/compat:v2_compat",
        "//machina/python/distribute:distribute_lib",
        "//machina/python/eager:backprop",
        "//machina/python/eager:def_function",
        "//machina/python/platform:client_testlib",
        "//machina/python/tpu:tpu_embedding_v1",
        "//machina/python/tpu:tpu_embedding_v2_utils",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tpu_py_strict_test(
    name = "tpu_initialization_test",
    srcs = [
        "tpu_initialization_test.py",
    ],
    disable_mlir_bridge = False,
    disable_tfrt = False,
    disable_v3_4chips = False,
    tags = ["no_oss"],
    deps = [
        "//machina/python/compat:v2_compat",
        "//machina/python/distribute/cluster_resolver:tpu_cluster_resolver_py",
        "//machina/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

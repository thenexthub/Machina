# Description:
#   Contains the Keras layers (internal TensorFlow version).

load("//machina:py.default.bzl", "py_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    # TODO(scottzhu): Remove non-keras deps from TF.
    default_visibility = [
        "//machina/python/distribute:__pkg__",
        "//machina/python/feature_column:__pkg__",
        "//machina/python/keras:__subpackages__",
        "//machina/python/trackable:__pkg__",
        "//machina/tools/pip_package:__pkg__",
        "//machina/tools/pip_package:__subpackages__",
        "//machina_models/official/projects/residual_mobilenet/modeling/backbones:__pkg__",
    ],
    licenses = ["notice"],
)

filegroup(
    name = "all_py_srcs",
    srcs = glob(["*.py"]),
    visibility = ["//machina/python/keras/google/private_tf_api_test:__pkg__"],
)

# A separate build for layers without serialization to avoid circular deps
# with feature column.
py_library(
    name = "layers",
    srcs = [
        "__init__.py",
        "serialization.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":advanced_activations",
        ":convolutional",
        ":convolutional_recurrent",
        ":core",
        ":dense_attention",
        ":embeddings",
        ":merge",
        ":pooling",
        ":recurrent",
        ":rnn_cell_wrapper_v2",
        "//machina/python/keras/engine",
        "//machina/python/keras/utils:tf_utils",
    ],
)

py_library(
    name = "advanced_activations",
    srcs = ["advanced_activations.py"],
    srcs_version = "PY3",
    deps = [
        "//machina/python/keras:backend",
        "//machina/python/keras:base_layer",
        "//machina/python/keras:constraints",
        "//machina/python/keras:regularizers",
        "//machina/python/keras/engine:input_spec",
        "//machina/python/keras/initializers",
        "//machina/python/keras/utils:tf_utils",
        "//machina/python/ops:math_ops",
        "//machina/python/util:tf_export",
    ],
)

py_library(
    name = "convolutional",
    srcs = ["convolutional.py"],
    srcs_version = "PY3",
    deps = [
        ":pooling",
        "//machina/python/eager:context",
        "//machina/python/framework:tensor_shape",
        "//machina/python/keras:activations",
        "//machina/python/keras:backend",
        "//machina/python/keras:base_layer",
        "//machina/python/keras:constraints",
        "//machina/python/keras:regularizers",
        "//machina/python/keras/engine:input_spec",
        "//machina/python/keras/initializers",
        "//machina/python/keras/utils:engine_utils",
        "//machina/python/keras/utils:tf_utils",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:nn",
        "//machina/python/ops:nn_ops",
        "//machina/python/util:tf_export",
    ],
)

py_library(
    name = "convolutional_recurrent",
    srcs = ["convolutional_recurrent.py"],
    srcs_version = "PY3",
    deps = [
        ":recurrent",
        "//machina/python/keras:activations",
        "//machina/python/keras:backend",
        "//machina/python/keras:base_layer",
        "//machina/python/keras:constraints",
        "//machina/python/keras:regularizers",
        "//machina/python/keras/engine:input_spec",
        "//machina/python/keras/initializers",
        "//machina/python/keras/utils:engine_utils",
        "//machina/python/keras/utils:generic_utils",
        "//machina/python/keras/utils:tf_utils",
        "//machina/python/ops:array_ops",
        "//machina/python/util:tf_export",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "core",
    srcs = ["core.py"],
    srcs_version = "PY3",
    deps = [
        "//machina/python/eager:backprop",
        "//machina/python/eager:context",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor_conversion",
        "//machina/python/framework:tensor_shape",
        "//machina/python/keras:activations",
        "//machina/python/keras:backend",
        "//machina/python/keras:base_layer",
        "//machina/python/keras:constraints",
        "//machina/python/keras:losses",
        "//machina/python/keras:regularizers",
        "//machina/python/keras/engine:input_spec",
        "//machina/python/keras/engine:keras_tensor",
        "//machina/python/keras/initializers",
        "//machina/python/keras/utils:engine_utils",
        "//machina/python/keras/utils:generic_utils",
        "//machina/python/keras/utils:tf_utils",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:nn",
        "//machina/python/ops:sparse_ops",
        "//machina/python/ops:standard_ops",
        "//machina/python/ops:variable_scope",
        "//machina/python/platform:tf_logging",
        "//machina/python/trackable:base",
        "//machina/python/util:dispatch",
        "//machina/python/util:nest",
        "//machina/python/util:tf_decorator_py",
        "//machina/python/util:tf_export",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "dense_attention",
    srcs = ["dense_attention.py"],
    srcs_version = "PY3",
    deps = [
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor_conversion",
        "//machina/python/framework:tensor_shape",
        "//machina/python/keras:backend",
        "//machina/python/keras:base_layer",
        "//machina/python/keras/utils:tf_utils",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:init_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:nn",
        "//machina/python/util:tf_export",
    ],
)

py_library(
    name = "embeddings",
    srcs = ["embeddings.py"],
    srcs_version = "PY3",
    deps = [
        "//machina/python/eager:context",
        "//machina/python/framework:ops",
        "//machina/python/keras:backend",
        "//machina/python/keras:constraints",
        "//machina/python/keras:regularizers",
        "//machina/python/keras/engine:base_layer",
        "//machina/python/keras/initializers",
        "//machina/python/keras/utils:tf_utils",
        "//machina/python/ops:embedding_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/util:tf_export",
    ],
)

py_library(
    name = "merge",
    srcs = ["merge.py"],
    srcs_version = "PY3",
    deps = [
        "//machina/python/keras:backend",
        "//machina/python/keras:base_layer",
        "//machina/python/keras/utils:tf_utils",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:nn",
        "//machina/python/util:tf_export",
    ],
)

py_library(
    name = "pooling",
    srcs = ["pooling.py"],
    srcs_version = "PY3",
    deps = [
        "//machina/python/framework:tensor_shape",
        "//machina/python/keras:backend",
        "//machina/python/keras:base_layer",
        "//machina/python/keras/engine:input_spec",
        "//machina/python/keras/utils:engine_utils",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:nn",
        "//machina/python/util:tf_export",
    ],
)

py_library(
    name = "recurrent",
    srcs = ["recurrent.py"],
    srcs_version = "PY3",
    deps = [
        "//machina/python/distribute:distribute_lib",
        "//machina/python/eager:context",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor_shape",
        "//machina/python/keras:activations",
        "//machina/python/keras:backend",
        "//machina/python/keras:base_layer",
        "//machina/python/keras:constraints",
        "//machina/python/keras:regularizers",
        "//machina/python/keras/engine:input_spec",
        "//machina/python/keras/initializers",
        "//machina/python/keras/utils:generic_utils",
        "//machina/python/keras/utils:tf_utils",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:cond",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:state_ops",
        "//machina/python/platform:tf_logging",
        "//machina/python/trackable:base",
        "//machina/python/trackable:data_structures",
        "//machina/python/util:nest",
        "//machina/python/util:tf_export",
        "//machina/tools/docs:doc_controls",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "rnn_cell_wrapper_v2",
    srcs = ["rnn_cell_wrapper_v2.py"],
    srcs_version = "PY3",
    deps = [
        ":recurrent",
        "//machina/python/keras/layers/legacy_rnn:rnn_cell_wrapper_impl",
        "//machina/python/util:deprecation",
        "//machina/python/util:tf_export",
    ],
)

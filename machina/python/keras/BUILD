# Description:
#   Contains the Keras API (internal TensorFlow version).

load("//machina:py.default.bzl", "py_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

filegroup(
    name = "all_py_srcs",
    srcs = glob(["*.py"]),
    visibility = [
        "//machina/python/keras/google/private_tf_api_test:__pkg__",
    ],
)

config_setting(
    name = "no_keras_py_deps",
    define_values = {"no_keras_py_deps": "true"},
    visibility = ["//visibility:public"],
)

py_library(
    name = "keras",
    srcs = [
        "__init__.py",
        "keras_parameterized.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":backend",
        ":engine",
        ":testing_utils",  # TODO(scottzhu): Stop exporting the test_utils after removing all the callers.
        "//machina/python/keras/distribute",
        "//machina/python/keras/layers",
        "//machina/python/keras/mixed_precision:mixed_precision_experimental",
        "//machina/python/keras/optimizer_v2",
        "//machina/python/keras/saving",
        "//machina/python/keras/utils",
        "//machina/python/platform:client_testlib",
        "//machina/python/saved_model",
        "//machina/python/training",
        "//machina/python/util:nest",
        "@pypi_h5py//:pkg",
    ],
)

py_library(
    name = "backend",
    srcs = ["backend.py"],
    srcs_version = "PY3",
    deps = [
        ":backend_config",
        "//machina/core:protos_all_py",
        "//machina/python/client",
        "//machina/python/client:session",
        "//machina/python/distribute:distribute_lib",
        "//machina/python/framework:composite_tensor",
        "//machina/python/framework:config",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/framework:sparse_tensor",
        "//machina/python/framework:tensor",
        "//machina/python/framework:tensor_conversion",
        "//machina/python/framework:tensor_shape",
        "//machina/python/keras/distribute:distribute_coordinator_utils",
        "//machina/python/keras/engine:keras_tensor",
        "//machina/python/keras/utils:control_flow_util",
        "//machina/python/keras/utils:object_identity",
        "//machina/python/keras/utils:tf_contextlib",
        "//machina/python/keras/utils:tf_inspect",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:check_ops",
        "//machina/python/ops:clip_ops",
        "//machina/python/ops:cond",
        "//machina/python/ops:control_flow_ops",
        "//machina/python/ops:ctc_ops",
        "//machina/python/ops:functional_ops",
        "//machina/python/ops:gradients",
        "//machina/python/ops:image_ops",
        "//machina/python/ops:init_ops",
        "//machina/python/ops:init_ops_v2",
        "//machina/python/ops:logging_ops",
        "//machina/python/ops:map_fn",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:nn",
        "//machina/python/ops:random_ops",
        "//machina/python/ops:sparse_ops",
        "//machina/python/ops:state_ops",
        "//machina/python/ops:tensor_array_grad",
        "//machina/python/ops:tensor_array_ops",
        "//machina/python/ops:variable_v1",
        "//machina/python/ops:variables",
        "//machina/python/ops:while_loop",
        "//machina/python/platform:tf_logging",
        "//machina/python/training:training_lib",
        "//machina/python/util:dispatch",
        "//machina/python/util:nest",
        "//machina/python/util:numpy_compat",
        "//machina/python/util:tf_export",
    ],
)

py_library(
    name = "backend_config",
    srcs = ["backend_config.py"],
    srcs_version = "PY3",
    deps = [
        "//machina/python/util:dispatch",
        "//machina/python/util:tf_export",
    ],
)

# TODO(scottzhu): Cleanup this target and point all the user to keras/engine.
py_library(
    name = "engine",
    srcs = [
        ":metrics",
        ":models",
    ],
    srcs_version = "PY3",
    deps = [
        "//machina/python/keras/engine",
    ],
)

py_library(
    name = "activations",
    srcs = [
        "activations.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":backend",
        "//machina/python/keras/layers:advanced_activations",
        "//machina/python/keras/utils:engine_utils",
        "//machina/python/util:dispatch",
        "//machina/python/util:tf_export",
    ],
)

# TODO(scottzhu): Cleanup this target and point all the user to keras/engine.
py_library(
    name = "base_layer",
    srcs = [],
    srcs_version = "PY3",
    deps = [
        "//machina/python/keras/engine:base_layer",
    ],
)

py_library(
    name = "callbacks",
    srcs = [
        "callbacks.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":backend",
        "//machina/python/framework:errors",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor",
        "//machina/python/keras/distribute:distributed_file_utils",
        "//machina/python/keras/distribute:worker_training_state",
        "//machina/python/keras/protobuf:projector_config_proto_py",
        "//machina/python/keras/utils:engine_utils",
        "//machina/python/keras/utils:mode_keys",
        "//machina/python/platform:gfile",
        "//machina/python/platform:tf_logging",
        "//machina/python/profiler:profiler_v2",
        "//machina/python/util:nest",
        "//machina/python/util:tf_export",
        "//machina/tools/docs:doc_controls",
    ],
)

py_library(
    name = "combinations",
    srcs = [
        "combinations.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":testing_utils",
        "//machina/python:tf2",
        "//machina/python/framework:combinations",
        "//machina/python/framework:test_combinations_lib",
    ],
)

py_library(
    name = "callbacks_v1",
    srcs = [
        "callbacks_v1.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":backend",
        "//machina/python/framework:errors",
        "//machina/python/keras/utils:engine_utils",
        "//machina/python/platform:tf_logging",
        "//machina/python/profiler:profiler_v2",
        "//machina/python/summary:summary_py",
    ],
)

py_library(
    name = "constraints",
    srcs = [
        "constraints.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":backend",
        "//machina/python/keras/utils:engine_utils",
        "//machina/python/ops:while_loop",
    ],
)

py_library(
    name = "losses",
    srcs = [
        "losses.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":backend",
        "//machina/python/framework:tensor_conversion",
        "//machina/python/keras/utils:engine_utils",
        "//machina/python/ops:cond",
        "//machina/python/util:dispatch",
    ],
)

py_library(
    name = "metrics",
    srcs = [
        "metrics.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":activations",
        ":backend",
        ":losses",
        "//machina/python/distribute:distribute_lib",
        "//machina/python/eager:context",
        "//machina/python/eager:def_function",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor_conversion",
        "//machina/python/framework:tensor_shape",
        "//machina/python/keras/distribute",
        "//machina/python/keras/engine:base_layer",
        "//machina/python/keras/engine:base_layer_utils",
        "//machina/python/keras/utils:generic_utils",
        "//machina/python/keras/utils:metrics_utils",
        "//machina/python/keras/utils:tf_utils",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:check_ops",
        "//machina/python/ops:confusion_matrix",
        "//machina/python/ops:init_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:nn",
        "//machina/python/ops:variables",
        "//machina/python/ops:weights_broadcast_ops",
        "//machina/python/ops/losses",
        "//machina/python/util:dispatch",
        "//machina/python/util:nest",
        "//machina/python/util:tf_export",
        "//machina/tools/docs:doc_controls",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "models",
    srcs = [
        "models.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":backend",
        ":metrics",
        ":optimizers",
        "//machina/python/keras/engine",
        "//machina/python/keras/engine:base_layer",
        "//machina/python/keras/saving",
        "//machina/python/keras/utils:generic_utils",
        "//machina/python/keras/utils:version_utils",
        "//machina/python/platform:tf_logging",
        "//machina/python/util:nest",
        "//machina/python/util:tf_export",
    ],
)

py_library(
    name = "optimizers",
    srcs = [
        "optimizer_v1.py",
        "optimizers.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":backend",
        "//machina/python/keras/optimizer_v2",
        "//machina/python/keras/utils:engine_utils",
        "//machina/python/util:nest",
        "//machina/python/util:tf_export",
    ],
)

py_library(
    name = "regularizers",
    srcs = [
        "regularizers.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":backend",
        "//machina/python/keras/utils:engine_utils",
    ],
)

# Internally urllib.request.urlretrieve library requires Google
# SSL context to be provided to work in python 3. This isn't needed in OSS.
# copybara:uncomment_begin(google-only)
# py_library(
#     name = "url_utils",
#     srcs = ["google/url_utils.py"],
#     srcs_version = "PY3",
#     deps = ["//pyglib/contrib/google_ssl"],
# )
# copybara:uncomment_end

py_library(
    name = "testing_utils",
    srcs = [
        "testing_utils.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":backend",
        ":models",
        "//machina/python:tf2",
        "//machina/python/eager:context",
        "//machina/python/framework:config",
        "//machina/python/framework:tensor_shape",
        "//machina/python/framework:tensor_spec",
        "//machina/python/framework:test_lib",
        "//machina/python/keras/engine:base_layer_utils",
        "//machina/python/keras/layers",
        "//machina/python/keras/optimizer_v2",
        "//machina/python/util:tf_decorator_py",
        "//third_party/py/numpy",
    ],
)

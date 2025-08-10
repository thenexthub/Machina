# DTensor Python API and libraries.

load("//machina:pytype.default.bzl", "pytype_strict_library")
load("//machina:strict.default.bzl", "py_strict_library")
load("//machina:machina.bzl", "tf_gen_op_wrapper_py")

default_visibility = [
    "//machina/dtensor:dtensor-internal",
    "//third_party/py/jax_tpu_embedding:__subpackages__",
]

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = default_visibility,
    licenses = ["notice"],
)

# -----------------------------------------------------------------------------
# A list of all modules linked into TensorFlow.

py_strict_library(
    name = "dtensor",
    srcs = ["__init__.py"],
    visibility = [
        "//machina/python:__pkg__",
        "//machina/python/tools/api/generator:__pkg__",
    ],
    deps = [
        ":accelerator_util",
        ":api",
        ":config",
        ":d_checkpoint",
        ":d_variable",
        ":input_util",
        ":layout",
        ":mesh_util",
        ":save_restore",
        ":tpu_util",
    ],
)

# -----------------------------------------------------------------------------
# Implementations of the public API.

pytype_strict_library(
    name = "api",
    srcs = ["api.py"],
    deps = [
        ":dtensor_device",
        ":gen_dtensor_ops",
        ":layout",
        "//machina/python/eager:context",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor",
        "//machina/python/util:deprecation",
        "//machina/python/util:tf_export",
    ],
)

pytype_strict_library(
    name = "config",
    srcs = ["config.py"],
    deps = [
        "//machina/python/eager:context",
        "//machina/python/framework:config",
        "//machina/python/framework:device",
        "//machina/python/util:tf_export",
    ],
)

tf_gen_op_wrapper_py(
    name = "gen_dtensor_ops",
    out = "gen_dtensor_ops.py",
    extra_py_deps = [
        "//machina/python:pywrap_tfe",
        "//machina/python/util:dispatch",
        "//machina/python/util:deprecation",
        "//machina/python/util:tf_export",
    ],
    py_lib_rule = py_strict_library,
    deps = [
        "//machina/dtensor/cc:dtensor_ops",
        "//machina/dtensor/cc:dtensor_tpu_ops",
    ],
)

pytype_strict_library(
    name = "layout",
    srcs = ["layout.py"],
    deps = [
        "//machina/dtensor/proto:layout_proto_py_pb2",
        "//machina/python:_pywrap_dtensor_device",
        "//machina/python/framework:device",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor",
        "//machina/python/util:tf_export",
        "//third_party/py/numpy",
    ],
)

pytype_strict_library(
    name = "d_random",
    srcs = ["d_random.py"],
    deps = [
        ":api",
        ":layout",
        "//machina/python/eager:context",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor_shape",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:math_ops_gen",
        "//machina/python/ops:shape_util",
        "//machina/python/ops:stateless_random_ops_gen",
    ],
)

pytype_strict_library(
    name = "d_variable",
    srcs = ["d_variable.py"],
    deps = [
        ":api",
        ":layout",
        "//machina/python/eager:context",
        "//machina/python/eager:def_function",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/framework:ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:resource_variable_ops",
        "//machina/python/trackable:base",
        "//machina/python/training/saving:saveable_object",
        "//machina/python/util:tf_export",
    ],
)

pytype_strict_library(
    name = "d_checkpoint",
    srcs = ["d_checkpoint.py"],
    deps = [
        ":api",
        ":d_variable",
        ":gen_dtensor_ops",
        ":layout",
        ":save_restore",
        "//machina/core:protos_all_py",
        "//machina/dtensor/proto:layout_proto_py_pb2",
        "//machina/python/checkpoint",
        "//machina/python/checkpoint:checkpoint_options",
        "//machina/python/checkpoint:graph_view",
        "//machina/python/checkpoint:restore",
        "//machina/python/eager:context",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:errors",
        "//machina/python/framework:ops",
        "//machina/python/ops:array_ops",
        "//machina/python/trackable:base",
        "//machina/python/trackable:data_structures",
        "//machina/python/training:py_checkpoint_reader",
        "//machina/python/training/saving:saveable_object",
        "//machina/python/training/saving:saveable_object_util",
        "//machina/python/util:deprecation",
        "//machina/python/util:nest",
        "//machina/python/util:tf_export",
    ],
)

pytype_strict_library(
    name = "numpy_util",
    srcs = ["numpy_util.py"],
    deps = [
        ":api",
        ":layout",
        "//machina/python/eager/polymorphic_function",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:array_ops_stack",
        "//machina/python/ops:sparse_ops",
        "//machina/python/ops:stateless_random_ops",
        "//machina/python/types:core",
        "//third_party/py/numpy",
    ],
)

pytype_strict_library(
    name = "save_restore",
    srcs = ["save_restore.py"],
    deps = [
        ":api",
        ":d_variable",
        ":gen_dtensor_ops",
        ":layout",
        ":mesh_util",
        "//machina/python/eager:context",
        "//machina/python/framework:errors",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor",
        "//machina/python/ops:io_ops",
        "//machina/python/ops:variables",
        "//machina/python/util:tf_export",
    ],
)

# -----------------------------------------------------------------------------
# The DTensor runtime.

pytype_strict_library(
    name = "dtensor_device",
    srcs = ["dtensor_device.py"],
    deps = [
        ":config",
        ":gen_dtensor_ops",
        ":layout",
        "//machina/core:protos_all_py",
        "//machina/python:_pywrap_dtensor_device",
        "//machina/python/eager:context",
        "//machina/python/eager:core",
        "//machina/python/framework:device",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/framework:sparse_tensor",
        "//machina/python/framework:tensor",
        "//machina/python/framework:tensor_util",
        "//machina/python/util:_pywrap_utils",
        "//third_party/py/numpy",
    ],
)

# -----------------------------------------------------------------------------
# Utilities.

pytype_strict_library(
    name = "mesh_util",
    srcs = ["mesh_util.py"],
    visibility = default_visibility + [
        "//machina/dtensor:dtensor-users",
    ],
    deps = [
        ":accelerator_util",
        ":api",
        ":config",
        ":layout",
        ":tpu_util",
        "//machina/python/eager:context",
        "//machina/python/framework:config",
        "//machina/python/framework:device",
        "//machina/python/framework:tfrt_utils",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/util:tf_export",
        "//third_party/py/numpy",
        "@absl_py//absl/flags",
        "@absl_py//absl/logging",
    ],
)

# TODO(b/245589661): Split accelerator_util to its module after
# The circular dependence is removed with dtensor_initialize_tpu_system.
pytype_strict_library(
    name = "tpu_util",
    srcs = [
        "accelerator_util.py",
        "tpu_util.py",
    ],
    visibility = default_visibility + [
        "//machina/dtensor:dtensor-users",
    ],
    deps = [
        ":config",
        ":dtensor_device",
        ":gen_dtensor_ops",
        ":layout",
        "//machina/core:protos_all_py",
        "//machina/python/eager:context",
        "//machina/python/eager:def_function",
        "//machina/python/framework:config",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:errors",
        "//machina/python/framework:ops",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/platform:remote_utils",
        "//machina/python/platform:tf_logging",
        "//machina/python/tpu:topology",
        "//machina/python/util:numpy_compat",
        "//machina/python/util:tf_export",
        "//third_party/py/numpy",
        "@absl_py//absl/logging",
    ],
)

pytype_strict_library(
    name = "heartbeat",
    srcs = ["heartbeat.py"],
    deps = [
        ":api",
        ":config",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:device",
        "//machina/python/framework:ops",
        "//machina/python/ops:collective_ops",
        "//machina/python/platform:tf_logging",
        "//third_party/py/numpy",
    ],
)

pytype_strict_library(
    name = "accelerator_util",
    srcs = [],
    deps = [
        ":tpu_util",
    ],
)

pytype_strict_library(
    name = "input_util",
    srcs = ["input_util.py"],
    deps = [
        ":api",
        ":config",
        ":layout",
        "//machina/python/data/experimental/ops:data_service_ops",
        "//machina/python/data/experimental/ops:distribute",
        "//machina/python/data/ops:dataset_ops",
        "//machina/python/data/ops:iterator_ops",
        "//machina/python/eager:context",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor",
        "//machina/python/framework:tensor_shape",
        "//machina/python/framework:tensor_spec",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/types:data",
        "//machina/python/util:nest",
        "//machina/python/util:tf_export",
    ],
)

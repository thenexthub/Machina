# Description:
#   Wrap NVIDIA TensorRT (http://developer.nvidia.com/tensorrt) with machina
#   and provide TensorRT operators and converter package.
#   APIs are meant to change over time.

load("//machina:strict.default.bzl", "py_strict_library")
load("//machina:machina.default.bzl", "cuda_py_strict_test")

# cuda_py_test and cuda_py_tests enable XLA tests by default. We can't
# combine XLA with TensorRT currently and should set
# xla_enable_strict_auto_jit to False to disable XLA tests.

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

exports_files(glob([
    "test/testdata/*",
]))

py_strict_library(
    name = "init_py",
    srcs = ["__init__.py"],
    deps = [
        ":tf_trt_integration_test_base",  # build_cleaner: keep
        ":trt_convert_py",
    ],
)

py_strict_library(
    name = "trt_convert_py",
    srcs = ["trt_convert.py"],
    deps = [
        ":utils",
        "//machina/compiler/tf2tensorrt:_pywrap_py_utils",
        "//machina/compiler/tf2tensorrt:trt_ops_loader",
        "//machina/core:protos_all_py",
        "//machina/python/client:session",
        "//machina/python/eager:context",
        "//machina/python/eager:wrap_function",
        "//machina/python/framework:convert_to_constants",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/framework:importer",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor",
        "//machina/python/grappler:tf_optimizer",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:resource_variable_ops_gen",
        "//machina/python/platform:tf_logging",
        "//machina/python/saved_model:builder",
        "//machina/python/saved_model:load",
        "//machina/python/saved_model:loader",
        "//machina/python/saved_model:save",
        "//machina/python/saved_model:signature_constants",
        "//machina/python/saved_model:tag_constants",
        "//machina/python/trackable:asset",
        "//machina/python/trackable:autotrackable",
        "//machina/python/trackable:resource",
        "//machina/python/training:saver",
        "//machina/python/util:deprecation",
        "//machina/python/util:lazy_loader",
        "//machina/python/util:nest",
        "//machina/python/util:tf_export",
        "//third_party/py/numpy",
        "@six_archive//:six",
    ],
)

py_strict_library(
    name = "utils",
    srcs = ["utils.py"],
    deps = [
        "//machina/compiler/tf2tensorrt:_pywrap_py_utils",
        "//machina/core:protos_all_py",
        "//machina/python/framework:dtypes",
        "@pypi_packaging//:pkg",
    ],
)

py_strict_library(
    name = "tf_trt_integration_test_base",
    deps = [
        ":trt_convert_py",
        ":utils",
        "//machina/compiler/tf2tensorrt:_pywrap_py_utils",
        "//machina/core:protos_all_py",
        "//machina/python/compiler/tensorrt/test:tf_trt_integration_test_base_srcs",
        "//machina/python/eager:def_function",
        "//machina/python/framework:config",
        "//machina/python/framework:graph_io",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor_spec",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/platform:client_testlib",
        "//machina/python/platform:tf_logging",
        "//machina/python/profiler:trace",
        "//machina/python/saved_model:builder",
        "//machina/python/saved_model:load",
        "//machina/python/saved_model:loader",
        "//machina/python/saved_model:save",
        "//machina/python/saved_model:signature_constants",
        "//machina/python/saved_model:signature_def_utils",
        "//machina/python/saved_model:tag_constants",
        "//machina/python/saved_model:utils",
        "//machina/python/tools:saved_model_utils",
        "//machina/python/trackable:autotrackable",
        "//machina/python/util:nest",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "trt_convert_test",
    srcs = ["trt_convert_test.py"],
    data = [
        "//machina/python/compiler/tensorrt/test:trt_convert_test_data",
    ],
    tags = [
        "no_cuda_on_cpu_tap",
        # TODO(b/297490791): Reenable after TensorRT regression has been fixed
        "no_oss",
        "no_pip",
        "nomac",
        # TODO(b/303453873): Re-enable tests once TensorRT has been updated
        "notap",
    ],
    xla_enable_strict_auto_jit = False,
    deps = [
        ":trt_convert_py",
        "//machina/compiler/tf2tensorrt:_pywrap_py_utils",
        "//machina/compiler/tf2tensorrt:trt_engine_instance_proto_py",
        "//machina/core:protos_all_py",
        "//machina/python/compiler/tensorrt/test:test_utils",
        "//machina/python/eager:def_function",
        "//machina/python/framework:config",
        "//machina/python/framework:convert_to_constants",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/framework:importer",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor_shape",
        "//machina/python/framework:tensor_spec",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:resource_variable_ops_gen",
        "//machina/python/ops:variables",
        "//machina/python/platform:client_testlib",
        "//machina/python/saved_model:builder",
        "//machina/python/saved_model:load",
        "//machina/python/saved_model:loader",
        "//machina/python/saved_model:save",
        "//machina/python/saved_model:save_options",
        "//machina/python/saved_model:signature_constants",
        "//machina/python/saved_model:signature_def_utils",
        "//machina/python/saved_model:tag_constants",
        "//machina/python/saved_model:utils",
        "//machina/python/tools:saved_model_utils",
        "//machina/python/trackable:autotrackable",
        "//machina/python/util:lazy_loader",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

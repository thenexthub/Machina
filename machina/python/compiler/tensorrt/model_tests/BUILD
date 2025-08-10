# Description:
#   Run sample models with TensorRT through TF-TRT bridge. Test TensorRT
#   numerics and latency.

load("//machina:strict.default.bzl", "py_strict_binary", "py_strict_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

exports_files(glob([
    "models/*",
]))

py_strict_library(
    name = "model_handler",
    srcs = ["model_handler.py"],
    deps = [
        "//machina/core:protos_all_py",
        "//machina/python/client:session",
        "//machina/python/compiler/tensorrt:trt_convert_py",
        "//machina/python/framework:convert_to_constants",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:importer",
        "//machina/python/framework:ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:random_ops",
        "//machina/python/platform:tf_logging",
        "//machina/python/saved_model:load",
        "//machina/python/saved_model:loader",
        "//machina/python/saved_model:signature_constants",
        "//machina/python/saved_model:tag_constants",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "result_analyzer",
    srcs = ["result_analyzer.py"],
    deps = [
        ":model_handler",
        "//machina/python/compiler/tensorrt:trt_convert_py",
        "//machina/python/platform:tf_logging",
        "//third_party/py/numpy",
    ],
)

py_strict_binary(
    name = "run_models",
    srcs = ["run_models.py"],
    data = ["sample_model/saved_model.pb"],
    deps = [
        ":model_handler",
        ":result_analyzer",
        "//machina/python/compiler/tensorrt:trt_convert_py",
        "//machina/python/eager:context",
        "//machina/python/framework:config",
        "//machina/python/framework:ops",
        "//machina/python/platform:client_testlib",
        "//machina/python/platform:gfile",
        "//machina/python/platform:tf_logging",
        "//machina/python/saved_model:signature_constants",
        "//machina/python/saved_model:tag_constants",
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
    ],
)

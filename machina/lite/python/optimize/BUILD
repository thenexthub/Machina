load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//machina:strict.default.bzl", "py_strict_library", "py_strict_test")
load("//machina:machina.default.bzl", "pybind_extension")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "calibration_wrapper_lib",
    srcs = ["calibration_wrapper.cc"],
    hdrs = ["calibration_wrapper.h"],
    deps = [
        "//machina/compiler/mlir/lite:offset_buffer",
        "//machina/compiler/mlir/lite/schema:schema_fbs_with_mutable",
        "//machina/lite:framework",
        "//machina/lite:shared_library",
        "//machina/lite:string_util",
        "//machina/lite/core:framework",
        "//machina/lite/core/c:common",
        "//machina/lite/core/kernels:builtin_ops",
        "//machina/lite/python/interpreter_wrapper:numpy",
        "//machina/lite/python/interpreter_wrapper:python_error_reporter",
        "//machina/lite/python/interpreter_wrapper:python_utils",
        "//machina/lite/tools/optimize:quantization_wrapper_utils",
        "//machina/lite/tools/optimize:quantize_model",
        "//machina/lite/tools/optimize/calibration:calibration_reader",
        "//machina/lite/tools/optimize/calibration:calibrator_lib",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:optional",
        "@local_xla//third_party/python_runtime:headers",  # buildcleaner: keep
    ],
)

pybind_extension(
    name = "_pywrap_machina_lite_calibration_wrapper",
    srcs = [
        "calibration_wrapper_pybind11.cc",
    ],
    hdrs = ["calibration_wrapper.h"],
    additional_stubgen_deps = [
        "//third_party/py/numpy:numpy",
    ],
    common_lib_packages = [
        "litert/python",
        "machina/lite/python",
    ],
    enable_stub_generation = True,
    link_in_framework = True,
    pytype_srcs = [
        "_pywrap_machina_lite_calibration_wrapper.pyi",
    ],
    wrap_py_init = True,
    deps = [
        ":calibration_wrapper_lib",
        "//machina/lite:framework",
        "//machina/lite/core:framework_stable",
        "//machina/python/lib/core:pybind11_lib",
        "@local_xla//third_party/python_runtime:headers",
        "@pybind11",
    ],
)

py_strict_library(
    name = "calibrator",
    srcs = [
        "calibrator.py",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":_pywrap_machina_lite_calibration_wrapper",  # buildcleaner: keep
        "//machina/lite/python:convert_phase",
        "//machina/lite/python:interpreter",
        "//machina/python/framework:dtypes",
        "//machina/python/util:lazy_loader",
        "//third_party/py/numpy",
    ],
)

py_strict_test(
    name = "calibrator_test",
    srcs = ["calibrator_test.py"],
    data = [
        ":test_data",
        "//machina/lite:testdata/multi_add.bin",
    ],
    tags = ["no_oss"],
    deps = [
        ":calibrator",
        "@absl_py//absl/testing:parameterized",
        #internal proto upb dep
        "//third_party/py/numpy",
        "//machina:machina_py_no_contrib",
        "//machina/lite/python:lite",
        "//machina/lite/python:schema_py",
        "//machina/lite/tools:flatbuffer_utils",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:test_lib",
        "//machina/python/platform:client_testlib",
        "//machina/python/platform:resource_loader",
    ],
)

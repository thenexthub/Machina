load("//machina:strict.default.bzl", "py_strict_test")
load("//machina:machina.default.bzl", "pybind_extension")
load("//machina/core/platform:build_config_root.bzl", "if_pywrap")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

pybind_extension(
    name = "format_converter_wrapper_pybind11",
    srcs = ["format_converter_wrapper_pybind11.cc"],
    common_lib_packages = [
        "litert/python",
        "machina/lite/python",
    ],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    data = [
        "format_converter_wrapper_pybind11.pyi",
    ],
    enable_stub_generation = True,
    features = ["-use_header_modules"],
    wrap_py_init = True,
    deps = [
        "//machina/lite/core/c:common",
        "//machina/lite/kernels/internal/utils:sparsity_format_converter",
        "@local_xla//third_party/python_runtime:headers",  # buildcleaner: keep
        "@pybind11",
    ],
)

py_strict_test(
    name = "format_converter_wrapper_pybind11_test",
    srcs = ["format_converter_wrapper_pybind11_test.py"],
    deps = [
        ":format_converter_wrapper_pybind11",
        "@absl_py//absl/testing:absltest",
        #internal proto upb dep
        "//third_party/py/numpy",
    ] + if_pywrap(
        if_true = ["//machina/lite/python:pywrap_tflite"],
    ),
)

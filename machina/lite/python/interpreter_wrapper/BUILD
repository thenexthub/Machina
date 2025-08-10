load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//machina:machina.default.bzl", "get_compatible_with_portable", "pybind_extension")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "numpy",
    srcs = ["numpy.cc"],
    hdrs = ["numpy.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//machina/lite:string_util",
        "//machina/lite/core/c:c_api_types",
        "//machina/lite/core/c:common",
        "//third_party/py/numpy:headers",
        "@local_xla//third_party/python_runtime:headers",  # buildcleaner: keep
    ],
)

cc_library(
    name = "interpreter_wrapper_lib",
    srcs = ["interpreter_wrapper.cc"],
    hdrs = [
        "interpreter_wrapper.h",
    ],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":numpy",
        ":python_error_reporter",
        ":python_utils",
        "//machina/lite:framework",
        "//machina/lite:shared_library",
        "//machina/lite:string_util",
        "//machina/lite:util",
        "//machina/lite/core:framework",
        "//machina/lite/core/api",
        "//machina/lite/core/c:common",
        "//machina/lite/core/kernels:builtin_ops",
        "//machina/lite/delegates/xnnpack:xnnpack_delegate",
        "//machina/lite/kernels:reference_ops",
        "//machina/lite/kernels/internal:compatibility",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@local_xla//third_party/python_runtime:headers",  # buildcleaner: keep
    ],
)

cc_library(
    name = "python_error_reporter",
    srcs = ["python_error_reporter.cc"],
    hdrs = ["python_error_reporter.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//machina/lite:stateful_error_reporter",
        "@local_xla//third_party/python_runtime:headers",  # buildcleaner: keep
    ],
)

cc_library(
    name = "python_utils",
    srcs = ["python_utils.cc"],
    hdrs = ["python_utils.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "@local_xla//third_party/python_runtime:headers",  # buildcleaner: keep
    ],
)

config_setting(
    name = "tflite_pip_with_flex",
    define_values = {
        "tflite_pip_with_flex": "true",
    },
)

pybind_extension(
    name = "_pywrap_machina_interpreter_wrapper",
    srcs = [
        "interpreter_wrapper_pybind11.cc",
    ],
    hdrs = ["interpreter_wrapper.h"],
    additional_stubgen_deps = [
        "//third_party/py/numpy:numpy",
    ],
    common_lib_packages = [
        "litert/python",
        "machina/lite/python",
    ],
    compatible_with = get_compatible_with_portable(),
    enable_stub_generation = True,
    link_in_framework = True,
    pytype_srcs = [
        "_pywrap_machina_interpreter_wrapper.pyi",
    ],
    wrap_py_init = True,
    deps = [
        ":interpreter_wrapper_lib",
        "//machina/lite:framework",
        "//machina/lite/core:framework_stable",
        "//machina/python/lib/core:pybind11_lib",
        "@local_xla//third_party/python_runtime:headers",
        "@pybind11",
    ] + select({
        ":tflite_pip_with_flex": ["//machina/lite/delegates/flex:delegate"],
        "//conditions:default": [],
    }),
)

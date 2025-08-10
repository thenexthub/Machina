load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//machina:pytype.default.bzl", "pytype_strict_library")
load("//machina:strict.default.bzl", "py_strict_test")
load("//machina:machina.default.bzl", "get_compatible_with_portable", "if_portable", "pybind_extension")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "metrics_wrapper_lib",
    srcs = if_portable(
        if_false = ["wrapper/metrics_wrapper_nonportable.cc"],
        if_true = ["wrapper/metrics_wrapper_portable.cc"],
    ),
    hdrs = ["wrapper/metrics_wrapper.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = ["//visibility:private"],
    deps = [
        "@local_xla//third_party/python_runtime:headers",
    ] + if_portable(
        if_false = [
            "//learning/brain/google/monitoring:metrics_exporter",
        ],
        if_true = [],
    ),
)

pybind_extension(
    name = "_pywrap_machina_lite_metrics_wrapper",
    srcs = ["wrapper/metrics_wrapper_pybind11.cc"],
    hdrs = ["wrapper/metrics_wrapper.h"],
    common_lib_packages = [
        "litert/python",
        "machina/lite/python",
    ],
    compatible_with = get_compatible_with_portable(),
    enable_stub_generation = True,
    pytype_srcs = [
        "_pywrap_machina_lite_metrics_wrapper.pyi",
    ],
    visibility = ["//visibility:public"],
    wrap_py_init = True,
    deps = [
        ":metrics_wrapper_lib",
        "//machina/python/lib/core:pybind11_lib",
        "@com_google_protobuf//:protobuf",
        "@local_xla//third_party/python_runtime:headers",
        "@pybind11",
    ],
)

pytype_strict_library(
    name = "metrics_wrapper",
    srcs = ["wrapper/metrics_wrapper.py"],
    deps = [
        ":_pywrap_machina_lite_metrics_wrapper",
        "//machina/compiler/mlir/lite/metrics:converter_error_data_proto_py",
        "//machina/compiler/mlir/lite/python:wrap_converter",
    ],
)

py_strict_test(
    name = "metrics_wrapper_test",
    srcs = ["wrapper/metrics_wrapper_test.py"],
    deps = [
        ":metrics_wrapper",
        #internal proto upb dep
        "//machina:machina_py",
        "//machina/lite/python:convert",
        "//machina/lite/python:lite",
        "//machina/python/framework:test_lib",
        "//machina/python/platform:client_testlib",
    ],
)

pytype_strict_library(
    name = "metrics_interface",
    srcs = ["metrics_interface.py"],
    compatible_with = get_compatible_with_portable(),
    visibility = ["//machina/lite/tools/pip_package:__subpackages__"],
)

genrule(
    name = "metrics_py_gen",
    srcs = if_portable(
        if_false = ["metrics_nonportable.py"],
        if_true = ["metrics_portable.py"],
    ),
    outs = ["metrics.py"],
    cmd = (
        "cat $(SRCS) > $(OUTS)"
    ),
    compatible_with = get_compatible_with_portable(),
)

pytype_strict_library(
    name = "metrics",
    srcs = ["metrics.py"],
    compatible_with = get_compatible_with_portable(),
    visibility = ["//machina/lite:__subpackages__"],
    deps = if_portable(
        if_false = [
            "//machina/compiler/mlir/lite/metrics:converter_error_data_proto_py",
            ":metrics_wrapper",
            "//machina/python/eager:monitoring",
        ],
        if_true = [],
    ) + [":metrics_interface"],
)

py_strict_test(
    name = "metrics_test",
    srcs = if_portable(
        if_false = ["metrics_nonportable_test.py"],
        if_true = ["metrics_portable_test.py"],
    ),
    data = [
        "//machina/lite/python/testdata/control_flow_v1_saved_model:saved_model.pb",
    ],
    main = if_portable(
        if_false = "metrics_nonportable_test.py",
        if_true = "metrics_portable_test.py",
    ),
    tags = ["notap"],  # TODO(b/373657707): Remove once we debug the failure.
    deps = [
        ":metrics",
        "@absl_py//absl/testing:parameterized",
        #internal proto upb dep
        "//third_party/py/numpy",
        "//machina:machina_py",
        "//machina/compiler/mlir/lite/metrics:converter_error_data_proto_py",
        "//machina/core:protos_all_py",
        "//machina/lite/python:convert",
        "//machina/lite/python:lite",
        "//machina/python/client:session",
        "//machina/python/eager:context",
        "//machina/python/eager:monitoring",
        "//machina/python/framework:convert_to_constants",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:importer",
        "//machina/python/framework:ops",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:array_ops_stack",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:string_ops",
        "//machina/python/ops/ragged:ragged_tensor",
        "//machina/python/platform:client_testlib",
        "//machina/python/platform:resource_loader",
        "//machina/python/saved_model",
        "//machina/python/trackable:autotrackable",
    ],
)

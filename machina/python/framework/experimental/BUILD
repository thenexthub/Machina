# Experimental Unified APIs for Eager and Graph modes.

load("//machina:strict.default.bzl", "py_strict_library")
load("//machina:machina.default.bzl", "cuda_py_strict_test", "tf_python_pybind_extension")
load("//machina/core/platform:build_config_root.bzl", "if_pywrap")
load(
    "//machina/tools/test:performance.bzl",
    "cuda_py_benchmark_test",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    licenses = ["notice"],
)

tf_python_pybind_extension(
    name = "_unified_api",
    srcs = ["unified_api.cc"],
    enable_stub_generation = True,
    features = ["-layering_check"],
    pytype_srcs = [
        "_unified_api.pyi",
    ],
    starlark_only = True,
    visibility = [
        "//machina/python:__pkg__",
    ],
    deps = [
        "//machina/c/eager:tfe_tensorhandle_internal",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core/lib/llvm_rtti",
        "//machina/core/platform:refcount",
        "//machina/python:unified_api_pywrap_required_headers",
        "//machina/python/lib/core:pybind11_lib",
        "@com_google_absl//absl/types:span",
        "@pybind11",
    ],
)

tf_python_pybind_extension(
    name = "_tape",
    srcs = ["tape.cc"],
    features = ["-layering_check"],
    starlark_only = True,
    visibility = [
        "//machina/python:__pkg__",
    ],
    deps = [
        "//machina/c/eager:tfe_tensorhandle_internal",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core/lib/llvm_rtti",
        "//machina/python:unified_api_pywrap_required_headers",
        "//machina/python/lib/core:pybind11_lib",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:span",
        "@pybind11",
    ] + if_pywrap(
        if_true = [
            "//machina/c/experimental/gradients/tape:tape_context",
            "//machina/c/experimental/gradients:math_grad",
            "//machina/c/experimental/gradients:nn_grad",
        ],
    ),
)

tf_python_pybind_extension(
    name = "_math_ops",
    srcs = ["math_ops.cc"],
    enable_stub_generation = True,
    pytype_srcs = [
        "_math_ops.pyi",
    ],
    starlark_only = True,
    visibility = [
        "//machina/python:__pkg__",
    ],
    deps = [
        "//machina/c/eager:tfe_tensorhandle_internal",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core/lib/llvm_rtti",
        "//machina/python:unified_api_pywrap_required_headers",
        "//machina/python/lib/core:pybind11_lib",
        "@com_google_absl//absl/types:span",
        "@pybind11",
    ] + if_pywrap(
        if_true = [
            "//machina/c/experimental/ops:math_ops",
        ],
    ),
)

tf_python_pybind_extension(
    name = "_nn_ops",
    srcs = ["nn_ops.cc"],
    enable_stub_generation = True,
    pytype_srcs = [
        "_nn_ops.pyi",
    ],
    starlark_only = True,
    visibility = [
        "//machina/python:__pkg__",
    ],
    deps = [
        "//machina/c/eager:tfe_tensorhandle_internal",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core/lib/llvm_rtti",
        "//machina/python:unified_api_pywrap_required_headers",
        "//machina/python/lib/core:pybind11_lib",
        "@com_google_absl//absl/types:span",
        "@pybind11",
    ] + if_pywrap(
        if_true = [
            "//machina/c/experimental/ops:nn_ops",
        ],
    ),
)

py_strict_library(
    name = "gradient_registry",
    srcs = ["gradient_registry.py"],
    deps = [":_tape"],
)

py_strict_library(
    name = "math_ops",
    srcs = ["math_ops.py"],
    deps = [
        ":_math_ops",
        ":context_stack",
    ],
)

py_strict_library(
    name = "nn_ops",
    srcs = ["nn_ops.py"],
    deps = [
        ":_nn_ops",
        ":context_stack",
    ],
)

py_strict_library(
    name = "tape",
    srcs = ["tape.py"],
    deps = [
        ":_tape",
        ":context_stack",
        ":gradient_registry",
        "//machina/python/util:nest",
    ],
)

py_strict_library(
    name = "def_function",
    srcs = ["def_function.py"],
    deps = [
        ":_unified_api",
        ":context_stack",
        "//machina/python/util:nest",
    ],
)

py_strict_library(
    name = "thread_local_stack",
    srcs = ["thread_local_stack.py"],
)

py_strict_library(
    name = "context_stack",
    srcs = ["context_stack.py"],
    deps = [":thread_local_stack"],
)

cuda_py_strict_test(
    name = "unified_api_test",
    size = "small",
    srcs = ["unified_api_test.py"],
    tags = [
        # Note(srbs): These python bindings are not
        # exported as part of the pip package yet so
        # this test is disabled.
        "no_pip",
        "no_windows",  # b/168218876
    ],
    deps = [
        ":_unified_api",
        ":context_stack",
        ":def_function",
        ":math_ops",
        ":nn_ops",
        ":tape",
        "//machina/python/eager:backprop",
        "//machina/python/eager:context",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:nn_grad",
        "//machina/python/ops:nn_ops",
        "//machina/python/ops:random_ops",
        "//machina/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_benchmark_test(
    name = "graph_building_test",
    size = "small",
    srcs = ["graph_building_test.py"],
    deps = [
        "//machina/core/config:flags_py",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:func_graph",
        "//machina/python/framework:ops",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops_gen",
        "//machina/python/ops:math_ops_gen",
        "//machina/python/ops:resource_variable_ops",
        "//machina/python/ops:resource_variable_ops_gen",
        "//machina/python/platform:client_testlib",
    ],
)

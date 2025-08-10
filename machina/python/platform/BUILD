# platform package

load("//machina:strict.default.bzl", "py_strict_library")
load(
    "//machina:machina.bzl",
    "if_oss",
)
load("//machina:machina.default.bzl", "pybind_extension", "tf_py_build_info_genrule", "tf_py_strict_test", "tf_python_pybind_extension")
load("//machina/core/platform:build_config.bzl", "pyx_library", "tf_additional_all_protos", "tf_additional_lib_deps", "tf_proto_library", "tf_protos_grappler")  # @unused

visibility = [
    "//machina:__subpackages__",
    "//machina/dtensor:dtensor-internal",
    # copybara:uncomment "//learning/brain/python/platform:__subpackages__",
    # copybara:uncomment "//learning/brain/contrib/eager/numlib/benchmarks/kumamon:__subpackages__",
    # copybara:uncomment "//learning/brain/mobile/lite/tooling/model_analyzer:__subpackages__",
    # copybara:uncomment "//machina_serving/model_servers:__subpackages__",
    # copybara:uncomment "//third_party/odml/model_customization/quantization:__subpackages__",
]

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = visibility,
    licenses = ["notice"],
)

tf_py_build_info_genrule(
    name = "py_build_info_gen",
    out = "build_info.py",
)

py_strict_library(
    name = "build_info",
    srcs = ["build_info.py"],
)

py_strict_library(
    name = "self_check",
    srcs = ["self_check.py"],
    deps = if_oss(
        [
            ":build_info",
            ":_pywrap_cpu_feature_guard",
        ],
    ),
)

py_strict_library(
    name = "benchmark",
    srcs = ["benchmark.py"],
    visibility = visibility + ["//machina:internal"],
    deps = [
        ":gfile",
        ":tf_logging",
        "//machina/core:protos_all_py",
        "//machina/python/client:timeline",
        "//machina/python/framework:ops",
        "//machina/python/util:tf_export",
        "//machina/python/util:tf_inspect",
        "@absl_py//absl:app",
    ],
)

py_strict_library(
    name = "analytics",
    srcs = ["analytics.py"],
)

py_strict_library(
    name = "device_context",
    srcs = ["device_context.py"],
    deps = [
        "//machina/python/framework:ops",
        "//machina/python/ops:control_flow_ops",
    ],
)

py_strict_library(
    name = "test",
    srcs = ["googletest.py"],
    # copybara:uncomment_begin(google-only)
    # visibility = visibility + ["//machina:internal"],
    # copybara:uncomment_end_and_comment_begin
    visibility = [
        "//visibility:public",
    ],
    # copybara:comment_end
    deps = [
        ":benchmark",
        ":flags",
        "//machina/python/util:tf_export",
    ] + if_oss([
        ":tf_logging",
        "@absl_py//absl:app",
        "@absl_py//absl/testing:absltest",
        "//machina/python/framework:errors",
        "//machina/python/util:tf_inspect",
    ]),
)

tf_py_strict_test(
    name = "resource_loader_test",
    size = "small",
    srcs = ["resource_loader_test.py"],
    data = [
        "resource_loader.py",
    ],
    tags = [
        "no_pip",
        "no_windows",
    ],
    deps = [
        ":resource_loader",
        ":test",
    ],
)

tf_py_strict_test(
    name = "sysconfig_test",
    size = "small",
    srcs = ["sysconfig_test.py"],
    data = [
        "sysconfig.py",
    ],
    tags = [
        "no_mac",  # TODO(b/259295275) re-enable after fixing sysconfig.get_path breakage
        "no_oss",  # TODO(b/259295275) re-enable after fixing sysconfig.get_path breakage
        "no_pip",
        "no_windows",
    ],
    deps = [
        ":client_testlib",
        ":sysconfig",
        ":test",
        "//machina:machina_py",
    ],
)

tf_py_strict_test(
    name = "flags_test",
    size = "small",
    srcs = ["flags_test.py"],
    tags = [
        "no_mac",  # TODO(b/259295275) re-enable after fixing sysconfig.get_path breakage
        "no_oss",  # TODO(b/263966250) re-enable after fixing sysconfig.get_path breakage
    ],
    deps = [
        ":client_testlib",
        ":flags",
        "@absl_py//absl/flags",
    ],
)

tf_py_strict_test(
    name = "stacktrace_handler_test",
    size = "small",
    srcs = ["stacktrace_handler_test.py"],
    tags = [
        "no_oss",  # TODO(b/263966250) re-enable after fixing sysconfig.get_path breakage
        "no_windows",
        "nomac",
        "nozapfhahn",
    ],
    deps = [
        ":client_testlib",
        ":tf_logging",
    ],
)

tf_py_strict_test(
    name = "app_test",
    size = "small",
    srcs = ["app_test.py"],
    tags = [
        "manual",
        "notap",
    ],
    deps = [
        ":app",
        ":flags",
    ],
)

tf_python_pybind_extension(
    name = "_pywrap_stacktrace_handler",
    srcs = ["stacktrace_handler_wrapper.cc"],
    hdrs = [
        "//machina/core/platform:stacktrace_handler_hdrs",
        "@local_tsl//tsl/platform:stacktrace_handler_hdrs",
    ],
    enable_stub_generation = True,
    pytype_srcs = [
        "_pywrap_stacktrace_handler.pyi",
    ],
    deps = [
        "@local_tsl//tsl/platform:stacktrace_handler_hdrs_lib",
        "@local_xla//third_party/python_runtime:headers",
        "@pybind11",
    ],
)

tf_python_pybind_extension(
    name = "_pywrap_cpu_feature_guard",
    srcs = ["cpu_feature_guard_wrapper.cc"],
    enable_stub_generation = True,
    pytype_srcs = [
        "_pywrap_cpu_feature_guard.pyi",
    ],
    deps = [
        "//machina/core/platform:cpu_feature_guard_hdr",  # Only depend on header to avoid ODR issues.
        "@local_xla//third_party/python_runtime:headers",
        "@pybind11",
    ],
)

py_strict_library(
    name = "client_testlib",
    srcs = ["test.py"],
    # copybara:uncomment_begin(google-only)
    # visibility = visibility + [
    # "//third_party/cloud_tpu/convergence_tools:__subpackages__",
    # "//third_party/mlperf:__subpackages__",
    # "//third_party/py/tf_slim:__subpackages__",
    # "//machina:internal",
    # "//machina_models:__subpackages__",
    # ],
    # copybara:uncomment_end_and_comment_begin
    visibility = [
        "//visibility:public",
    ],
    # copybara:comment_end
    deps = [
        ":test",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:gradient_checker",
        "//machina/python/util:tf_export",
    ],
)

py_strict_library(
    name = "app",
    srcs = ["app.py"],
    deps = [
        ":flags",
        "//machina/python/util:_pywrap_util_port",
        "//machina/python/util:tf_export",
        "@absl_py//absl:app",
    ],
)

py_strict_library(
    name = "sysconfig",
    srcs = ["sysconfig.py"],
    deps = [
        ":build_info",
        "//machina/python/client:pywrap_tf_session",
        "//machina/python/framework:versions",
        "//machina/python/util:tf_export",
    ],
)

py_strict_library(
    name = "__init__",
    srcs = ["__init__.py"],
    deps = [
    ],
)

py_strict_library(
    name = "control_imports",
    srcs = ["control_imports.py"],
    deps = [
    ],
)

py_strict_library(
    name = "parameterized",
    srcs = ["parameterized.py"],
)

py_strict_library(
    name = "remote_utils",
    srcs = ["remote_utils.py"],
    deps = [
    ],
)

py_strict_library(
    name = "gfile",
    srcs = ["gfile.py"],
    # copybara:uncomment_begin(google-only)
    # visibility = visibility,
    # copybara:uncomment_end_and_comment_begin
    visibility = [
        "//visibility:public",
    ],
    # copybara:comment_end
    deps = [
        "//machina/python/lib/io:file_io",
        "//machina/python/util:deprecation",
        "//machina/python/util:tf_export",
    ],
)

py_strict_library(
    name = "tf_logging",
    srcs = ["tf_logging.py"],
    # copybara:uncomment_begin(google-only)
    # visibility = visibility + [
    # "//learning/brain/contrib/learn:__pkg__",
    # "//learning/brain/mobile/lite/lstm:__pkg__",
    # "//third_party/py/tf_slim/training:__pkg__",
    # ],
    # copybara:uncomment_end_and_comment_begin
    visibility = [
        "//visibility:public",
    ],
    # copybara:comment_end
    deps = [
        "//machina/python/util:tf_export",
        "@absl_py//absl/logging",
    ],
)

py_strict_library(
    name = "flags",
    srcs = ["flags.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//machina/python/util:tf_decorator_py",
        "@absl_py//absl/flags",
    ],
)

py_strict_library(
    name = "resource_loader",
    srcs = ["resource_loader.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":tf_logging",
        "//machina/python/util:tf_export",
        "//machina/python/util:tf_inspect",
    ] + if_oss([
        "@rules_python//python/runfiles",
    ]),
)

tf_py_strict_test(
    name = "build_info_test",
    size = "small",
    srcs = ["build_info_test.py"],
    main = "build_info_test.py",
    tags = [
        "no_pip",
        "notap",
    ],
    deps = [
        ":build_info",
        ":client_testlib",
        "//machina/compiler/tf2tensorrt:_pywrap_py_utils",
    ],
)

tf_py_strict_test(
    name = "benchmark_test",
    size = "small",
    srcs = ["benchmark_test.py"],
    main = "benchmark_test.py",
    tags = [
        "no_pip",
    ],
    deps = [
        ":benchmark",
        ":client_testlib",
        "//machina/core:protos_all_py",
    ],
)

pybind_extension(
    name = "_pywrap_tf2",
    srcs = ["enable_tf2.cc"],
    hdrs = ["//machina/core/platform:enable_tf2_hdr"],
    enable_stub_generation = True,
    pytype_srcs = [
        "_pywrap_tf2.pyi",
    ],
    deps = [
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core/platform:enable_tf2_utils",
        "@pybind11",
    ],
)

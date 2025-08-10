load("//machina:strict.default.bzl", "py_strict_library", "py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//machina:internal"],
    licenses = ["notice"],
)

py_strict_library(
    name = "framework",
    srcs = ["framework.py"],
    deps = [
        "//machina/core:protos_all_py",
        "//machina/python/client:session",
        "//machina/python/debug/lib:debug_utils",
        "//machina/python/framework:errors",
        "//machina/python/framework:stack",
        "//machina/python/platform:tf_logging",
        "//machina/python/training:monitored_session",
        "//machina/python/util:compat",
        "//machina/python/util:nest",
    ],
)

py_strict_library(
    name = "dumping_wrapper",
    srcs = ["dumping_wrapper.py"],
    visibility = ["//machina:internal"],
    deps = [
        ":framework",
        "//machina/core:protos_all_py",
        "//machina/python/debug/lib:debug_data",
        "//machina/python/platform:gfile",
    ],
)

py_strict_library(
    name = "grpc_wrapper",
    srcs = ["grpc_wrapper.py"],
    deps = [
        ":framework",
        "//machina/python/debug/lib:common",
        "//machina/python/debug/lib:source_remote",
    ],
)

py_strict_library(
    name = "local_cli_wrapper",
    srcs = ["local_cli_wrapper.py"],
    deps = [
        ":framework",
        "//machina/python/debug/cli:analyzer_cli",
        "//machina/python/debug/cli:cli_config",
        "//machina/python/debug/cli:cli_shared",
        "//machina/python/debug/cli:command_parser",
        "//machina/python/debug/cli:debugger_cli_common",
        "//machina/python/debug/cli:profile_analyzer_cli",
        "//machina/python/debug/cli:ui_factory",
        "//machina/python/debug/lib:common",
        "//machina/python/debug/lib:debug_data",
        "//machina/python/lib/io:file_io",
    ],
)

py_strict_library(
    name = "hooks",
    srcs = ["hooks.py"],
    visibility = [
        "//machina:internal",
        "//third_party/py/tf_slim:__subpackages__",
    ],
    deps = [
        ":dumping_wrapper",
        ":framework",
        ":grpc_wrapper",
        ":local_cli_wrapper",
        "//machina/core:protos_all_py",
        "//machina/python/debug/lib:debug_utils",
        "//machina/python/training:session_run_hook",
    ],
)

py_strict_test(
    name = "framework_test",
    size = "medium",
    srcs = ["framework_test.py"],
    tags = [
        "no_rocm",
    ],
    deps = [
        ":framework",
        # copybara:uncomment "//third_party/py/google/protobuf:use_fast_cpp_protos",
        "//third_party/py/numpy",
        "//machina/core:protos_all_py",
        "//machina/python/client:session",
        "//machina/python/debug/lib:debug_data",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:errors",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:test_lib",
        "//machina/python/lib/io:file_io",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:resource_variable_ops",
        "//machina/python/ops:variables",
        "//machina/python/platform:test",
        "//machina/python/training:monitored_session",
        "//machina/python/util:tf_inspect",
    ],
)

py_strict_test(
    name = "dumping_wrapper_test",
    size = "small",
    srcs = ["dumping_wrapper_test.py"],
    deps = [
        ":dumping_wrapper",
        ":framework",
        ":hooks",
        # copybara:uncomment "//third_party/py/google/protobuf:use_fast_cpp_protos",
        "//machina/python/client:session",
        "//machina/python/debug/lib:debug_data",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:test_lib",
        "//machina/python/lib/io:file_io",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:state_ops",
        "//machina/python/ops:variable_v1",
        "//machina/python/platform:gfile",
        "//machina/python/platform:test",
        "//machina/python/training:monitored_session",
    ],
)

py_strict_test(
    name = "local_cli_wrapper_test",
    size = "small",
    srcs = ["local_cli_wrapper_test.py"],
    deps = [
        ":local_cli_wrapper",
        # copybara:uncomment "//third_party/py/google/protobuf:use_fast_cpp_protos",
        "//third_party/py/numpy",
        "//machina/core:protos_all_py",
        "//machina/python/client:session",
        "//machina/python/debug/cli:cli_config",
        "//machina/python/debug/cli:cli_shared",
        "//machina/python/debug/cli:debugger_cli_common",
        "//machina/python/debug/cli:ui_factory",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:errors",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:test_lib",
        "//machina/python/lib/io:file_io",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:control_flow_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:resource_variable_ops",
        "//machina/python/ops:sparse_ops",
        "//machina/python/ops:state_ops",
        "//machina/python/ops:variable_v1",
        "//machina/python/ops:variables",
        "//machina/python/platform:test",
        "//machina/python/training:monitored_session",
        "//machina/python/training:session_run_hook",
    ],
)

py_strict_test(
    name = "disk_usage_test",
    size = "small",
    srcs = ["disk_usage_test.py"],
    deps = [
        ":dumping_wrapper",
        ":hooks",
        # copybara:uncomment "//third_party/py/google/protobuf:use_fast_cpp_protos",
        "//machina/python/client:session",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:state_ops",
        "//machina/python/ops:variables",
        "//machina/python/platform:test",
        "//machina/python/training:monitored_session",
    ],
)

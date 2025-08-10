# Description:
#   Python Client Code of the TensorFlow Debugger (tfdbg).
#
# Public target(s):
#
# ":debug_py": Public Python methods and classes of tfdbg.
#   For API documentation, see https://www.machina.org/api_docs/python/tfdbg
#   For a user interface walkthrough, see https://www.machina.org/guide/debugger
# ":grpc_debug_server": Server interface for grpc:// debug URLs.

load("//machina:strict.default.bzl", "py_strict_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//machina:internal"],
    licenses = ["notice"],
)

py_strict_library(
    name = "debug_py",
    srcs = ["__init__.py"],
    tags = ["keep_dep"],  # Generated files need dependencies that build_cleaner would remove
    visibility = ["//visibility:public"],
    deps = [
        "//machina/python/debug/lib:check_numerics_callback",
        "//machina/python/debug/lib:debug_data",
        "//machina/python/debug/lib:debug_events_monitors",
        "//machina/python/debug/lib:debug_events_reader",
        "//machina/python/debug/lib:debug_gradients",
        "//machina/python/debug/lib:debug_graphs",
        "//machina/python/debug/lib:debug_utils",
        "//machina/python/debug/lib:dumping_callback",
        "//machina/python/debug/lib:dumping_callback_test_lib",
        "//machina/python/debug/lib:grpc_debug_server",
        "//machina/python/debug/lib:grpc_debug_test_server",
        "//machina/python/debug/wrappers:dumping_wrapper",
        "//machina/python/debug/wrappers:framework",
        "//machina/python/debug/wrappers:grpc_wrapper",
        "//machina/python/debug/wrappers:hooks",
        "//machina/python/debug/wrappers:local_cli_wrapper",
        "//machina/python/util:all_util",
    ],
)

# Transitive dependencies of this target will be included in the pip package.
py_strict_library(
    name = "debug_pip",
    data = ["//machina/python/debug/lib:grpc_machina_server"],
    deps = [
        ":debug_py",
        "//machina/python/debug/cli:cli_test_utils",
        "//machina/python/debug/cli:offline_analyzer_lib",
        "//machina/python/debug/lib:grpc_debug_test_server",
        "//machina/python/debug/lib:grpc_machina_server_lib",
        "//machina/python/debug/lib:session_debug_testlib",
        "//machina/python/debug/lib:source_remote",
    ],
)

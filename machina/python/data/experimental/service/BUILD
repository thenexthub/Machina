load("//machina:strict.default.bzl", "py_strict_library")
load("//machina:machina.default.bzl", "tf_py_strict_test", "tf_python_pybind_extension")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//machina:internal"],
    licenses = ["notice"],
)

tf_python_pybind_extension(
    name = "_pywrap_server_lib",
    srcs = ["server_lib_wrapper.cc"],
    enable_stub_generation = True,
    pytype_srcs = [
        "_pywrap_server_lib.pyi",
    ],
    deps = [
        "//machina/core:lib",
        "//machina/core/data/service:common_proto_cc",
        "//machina/core/data/service:dispatcher_client",
        "//machina/core/data/service:grpc_util",
        "//machina/core/data/service:server_lib",
        "//machina/core/data/service:server_lib_headers_lib",
        "//machina/core/protobuf:for_core_protos_cc",
        "//machina/python/lib/core:pybind11_lib",
        "//machina/python/lib/core:pybind11_status",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@local_xla//third_party/python_runtime:headers",
        "@pybind11",
        "@pybind11_protobuf//pybind11_protobuf:native_proto_caster",
    ],
)

py_strict_library(
    name = "server_lib",
    srcs = ["server_lib.py"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        ":_pywrap_server_lib",
        ":_pywrap_utils_exp",
        "//machina/core:protos_all_py",
        "//machina/python:pywrap_machina",
        "//machina/python/util:tf_export",
    ],
)

tf_py_strict_test(
    name = "server_lib_test",
    srcs = ["server_lib_test.py"],
    deps = [
        ":server_lib",
        "//machina/python/framework:errors",
        "//machina/python/platform:client_testlib",
        "//machina/python/profiler:profiler_client",
    ],
)

tf_python_pybind_extension(
    name = "_pywrap_utils_exp",
    srcs = ["utils_wrapper.cc"],
    enable_stub_generation = True,
    pytype_srcs = [
        "_pywrap_utils_exp.pyi",
    ],
    deps = [
        "//machina/core/data/service:py_utils",
        "//machina/python/lib/core:pybind11_lib",
        "@local_xla//third_party/python_runtime:headers",
        "@pybind11",
    ],
)

tf_python_pybind_extension(
    name = "_pywrap_snapshot_utils",
    srcs = ["snapshot_utils_wrapper.cc"],
    enable_stub_generation = True,
    pytype_srcs = [
        "_pywrap_snapshot_utils.pyi",
    ],
    deps = [
        "//machina/core/data/service/snapshot:path_utils",
        "//machina/python/lib/core:pybind11_lib",
        "@local_xla//third_party/python_runtime:headers",
        "@pybind11",
    ],
)

py_strict_library(
    name = "service",
    srcs = ["__init__.py"],
    deps = [
        ":server_lib",
        "//machina/python/data/experimental/ops:data_service_ops",
    ],
)

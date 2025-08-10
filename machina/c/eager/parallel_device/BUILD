load(
    "//machina:machina.bzl",
    "tf_cc_test",
)
load("//machina:machina.default.bzl", "filegroup")
load("//machina/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    licenses = ["notice"],
)

# Currently pybind extension shared objects must use only C API headers since
# the C API has static initializers duplicated in the Python bindings. So we
# need a second rule that omits .cc files, in
# machina/python:_pywrap_parallel_device.
filegroup(
    name = "lib_headers",
    srcs = ["parallel_device_lib.h"],
)

filegroup(
    name = "lib_sources",
    srcs = ["parallel_device_lib.cc"],
)

filegroup(
    name = "device_headers",
    srcs = ["parallel_device.h"],
)

filegroup(
    name = "device_sources",
    srcs = ["parallel_device.cc"],
)

filegroup(
    name = "headers",
    srcs = [
        ":device_headers",
        ":lib_headers",
    ],
    visibility = ["//machina/python:__pkg__"],
)

filegroup(
    name = "sources",
    srcs = [
        ":device_sources",
        ":lib_sources",
    ],
    visibility = ["//machina/python:__pkg__"],
)

cc_library(
    name = "parallel_device",
    srcs = [":device_sources"],
    hdrs = [":device_headers"],
    visibility = ["//visibility:private"],
    deps = [
        ":parallel_device_lib",
        "//machina/c:c_api",
        "//machina/c:tf_status_helper",
        "//machina/c/eager:c_api",
        "//machina/c/eager:c_api_experimental",
        "//machina/c/eager:tfe_tensorhandle_internal",
        "//machina/core/platform:status",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:variant",
    ],
)

cc_library(
    name = "parallel_device_lib",
    srcs = [":lib_sources"],
    hdrs = [":lib_headers"],
    visibility = ["//machina:internal"],
    deps = [
        "//machina/c:c_api",
        "//machina/c:safe_ptr",
        "//machina/c:tf_status_internal",
        "//machina/c/eager:c_api",
        "//machina/c/eager:c_api_experimental",
        "//machina/c/eager:immediate_execution_tensor_handle",
        "//machina/c/eager:tfe_cancellation_manager_internal",
        "//machina/c/eager:tfe_op_internal",
        "//machina/c/eager:tfe_tensorhandle_internal",
        "//machina/core:framework",
        "//machina/core:lib",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
        "@com_google_absl//absl/types:variant",
    ],
)

tf_cc_test(
    name = "parallel_device_lib_test",
    srcs = ["parallel_device_lib_test.cc"],
    deps = [
        ":parallel_device_lib",
        ":parallel_device_testlib",
        "//machina/c:c_api",
        "//machina/c:c_api_experimental",
        "//machina/c/eager:c_api",
        "//machina/c/eager:c_api_experimental",
        "//machina/c/eager:tfe_context_internal",
        "//machina/core:framework",
        "//machina/core:protos_all_cc",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core/common_runtime/eager:context",
        "//machina/core/platform:status",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "parallel_device_testlib",
    testonly = 1,
    srcs = ["parallel_device_testlib.cc"],
    hdrs = ["parallel_device_testlib.h"],
    deps = [
        ":parallel_device",
        ":parallel_device_lib",
        "//machina/c:c_api",
        "//machina/c:c_api_experimental",
        "//machina/c/eager:c_api",
        "//machina/c/eager:c_api_experimental",
        "//machina/core:framework_types_hdr",
        "//machina/core:test",
        "//machina/core:test_main",
    ],
)

tf_cc_test(
    name = "parallel_device_test",
    srcs = ["parallel_device_test.cc"],
    deps = [
        ":parallel_device_lib",
        ":parallel_device_testlib",
        "//machina/c:c_api",
        "//machina/c:c_api_experimental",
        "//machina/c:tf_status_internal",
        "//machina/c/eager:c_api",
        "//machina/c/eager:c_api_experimental",
        "//machina/c/eager:immediate_execution_tensor_handle",
        "//machina/c/eager:tfe_tensorhandle_internal",
        "//machina/core:test",
        "//machina/core:test_main",
        "@com_google_googletest//:gtest_main",
    ],
)

tf_cc_test(
    name = "parallel_device_remote_test",
    srcs = ["parallel_device_remote_test.cc"],
    # TODO(b/136478427): Enable global heap checking when servers shut down
    # cleanly.
    args = ["--heap_check="],
    deps = [
        ":parallel_device_lib",
        ":parallel_device_testlib",
        "//machina/c:c_api",
        "//machina/c:c_api_experimental",
        "//machina/c/eager:c_api",
        "//machina/c/eager:c_api_experimental",
        "//machina/core:protos_all_cc",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core/distributed_runtime:master_env",
        "//machina/core/distributed_runtime/rpc:grpc_server_lib",
        "//machina/core/platform:strcat",
        "@com_google_absl//absl/log",
    ],
)

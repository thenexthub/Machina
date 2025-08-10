# This package contains written convenience helpers for Eager Operations
# used by SavedModel. Once we autogenerate C++ Eager Op wrappers, we can remove these.
load(
    "//machina:machina.bzl",
    "tf_cc_test",
)
load("//machina/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = [
        # Restricting visibility for now
        "//machina/c/experimental/saved_model/core:__subpackages__",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "restore_ops",
    srcs = [
        "restore_ops.cc",
    ],
    hdrs = [
        "restore_ops.h",
    ],
    deps = [
        "//machina/c:tensor_interface",
        "//machina/c/eager:abstract_tensor_handle",
        "//machina/c/eager:immediate_execution_context",
        "//machina/c/eager:immediate_execution_operation",
        "//machina/c/eager:immediate_execution_tensor_handle",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core/lib/llvm_rtti",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "variable_ops",
    srcs = [
        "variable_ops.cc",
    ],
    hdrs = [
        "variable_ops.h",
    ],
    deps = [
        "//machina/c/eager:abstract_tensor_handle",
        "//machina/c/eager:immediate_execution_context",
        "//machina/c/eager:immediate_execution_operation",
        "//machina/c/eager:immediate_execution_tensor_handle",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core/lib/llvm_rtti",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:span",
    ],
)

tf_cc_test(
    name = "restore_ops_test",
    srcs = [
        "restore_ops_test.cc",
    ],
    data = [
        "//machina/cc/saved_model:saved_model_half_plus_two",
    ],
    deps = [
        ":restore_ops",
        "//machina/c:tensor_interface",
        "//machina/c/eager:immediate_execution_tensor_handle",
        "//machina/c/experimental/saved_model/core:test_utils",
        "//machina/cc/saved_model:constants",
        "//machina/core:all_kernels",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core/common_runtime:core_cpu_lib",
        "//machina/core/common_runtime/eager:context",
        "//machina/core/common_runtime/eager:core",
        "@com_google_absl//absl/status",
    ],
)

tf_cc_test(
    name = "variable_ops_test",
    srcs = [
        "variable_ops_test.cc",
    ],
    deps = [
        ":variable_ops",
        "//machina/c:tensor_interface",
        "//machina/c/eager:immediate_execution_tensor_handle",
        "//machina/c/experimental/saved_model/core:test_utils",
        "//machina/core:all_kernels",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core/common_runtime:core_cpu_lib",
        "//machina/core/common_runtime/eager:context",
        "//machina/core/common_runtime/eager:core",
        "@com_google_absl//absl/status",
    ],
)

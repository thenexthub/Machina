load("//machina/core/platform:rules_cc.bzl", "cc_library")

# This package contains classes corresponding to Revived SavedObjectGraph types
# used by SavedModel. See https://cs.opensource.google/machina/machina/+/c575e2ba93c442121d98d3f125d83fed1339924d:machina/core/protobuf/saved_object_graph.proto;l=56-62
package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = [
        # Restricting visibility for now
        "//machina/c/experimental/saved_model/core:__pkg__",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "asset",
    srcs = [
        "asset.cc",
    ],
    hdrs = [
        "asset.h",
    ],
    deps = [
        ":tensorhandle_convertible",
        "//machina/c:tensor_interface",
        "//machina/c/eager:immediate_execution_context",
        "//machina/c/eager:immediate_execution_tensor_handle",
        "//machina/cc/saved_model:constants",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "constant",
    srcs = [
        "constant.cc",
    ],
    hdrs = [
        "constant.h",
    ],
    deps = [
        ":tensorhandle_convertible",
        "//machina/c:tensor_interface",
        "//machina/c/eager:immediate_execution_context",
        "//machina/c/eager:immediate_execution_tensor_handle",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "@com_google_absl//absl/status",
    ],
)

cc_library(
    name = "flat_tensor_function",
    srcs = [
        "flat_tensor_function.cc",
    ],
    hdrs = [
        "flat_tensor_function.h",
    ],
    deps = [
        "//machina/c/eager:abstract_tensor_handle",
        "//machina/c/eager:immediate_execution_context",
        "//machina/c/eager:immediate_execution_operation",
        "//machina/c/eager:immediate_execution_tensor_handle",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "partially_revived_objects",
    srcs = [
        "partially_revived_objects.cc",
    ],
    hdrs = [
        "partially_revived_objects.h",
    ],
    deps = [
        ":asset",
        ":constant",
        ":restored_resource",
        ":restored_resource_revival_state",
        ":revived_objects",
        ":tf_concrete_function",
        ":tf_concrete_function_revival_state",
        ":tf_signature_def_function",
        ":tf_signature_def_function_revival_state",
        ":variable",
        "//machina/c/eager:abstract_tensor_handle",
        "//machina/c/eager:immediate_execution_context",
        "//machina/c/eager:immediate_execution_operation",
        "//machina/c/eager:immediate_execution_tensor_handle",
        "//machina/c/experimental/saved_model/core:signature_def_function_metadata",
        "//machina/c/experimental/saved_model/core:tensor_spec",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core/lib/llvm_rtti",
        "//machina/core/platform:hash",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "restored_resource",
    srcs = [
        "restored_resource.cc",
    ],
    hdrs = [
        "restored_resource.h",
    ],
    deps = [
        ":tensorhandle_convertible",
        ":tf_concrete_function",
        "//machina/c/eager:abstract_tensor_handle",
        "//machina/c/eager:immediate_execution_operation",
        "//machina/c/eager:immediate_execution_tensor_handle",
        "//machina/core:lib",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "restored_resource_revival_state",
    hdrs = [
        "restored_resource_revival_state.h",
    ],
    deps = [
        ":tf_concrete_function_revival_state",
        "//machina/c/eager:immediate_execution_tensor_handle",
    ],
)

cc_library(
    name = "revived_objects",
    hdrs = [
        "revived_objects.h",
    ],
    deps = [
        ":asset",
        ":constant",
        ":restored_resource",
        ":tf_concrete_function",
        ":tf_signature_def_function",
        ":variable",
        "//machina/core:lib",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)

cc_library(
    name = "variable",
    srcs = [
        "variable.cc",
    ],
    hdrs = [
        "variable.h",
    ],
    deps = [
        ":tensorhandle_convertible",
        "//machina/c/eager:immediate_execution_context",
        "//machina/c/eager:immediate_execution_tensor_handle",
        "//machina/c/experimental/saved_model/core/ops:variable_ops",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core/common_runtime/eager:context",
        "//machina/core/common_runtime/eager:tensor_handle",
        "//machina/core/lib/llvm_rtti",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:optional",
    ],
)

cc_library(
    name = "tensorhandle_convertible",
    hdrs = [
        "tensorhandle_convertible.h",
    ],
    deps = [
        "//machina/c/eager:immediate_execution_tensor_handle",
    ],
)

cc_library(
    name = "tf_concrete_function",
    srcs = [
        "tf_concrete_function.cc",
    ],
    hdrs = [
        "tf_concrete_function.h",
    ],
    deps = [
        ":flat_tensor_function",
        "//machina/c/eager:abstract_tensor_handle",
        "//machina/c/eager:immediate_execution_context",
        "//machina/c/eager:immediate_execution_operation",
        "//machina/c/eager:immediate_execution_tensor_handle",
        "//machina/c/experimental/saved_model/core:concrete_function",
        "//machina/c/experimental/saved_model/core:function_metadata",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "tf_concrete_function_revival_state",
    hdrs = [
        "tf_concrete_function_revival_state.h",
    ],
    deps = [
        ":tf_concrete_function",
        "//machina/c/eager:immediate_execution_context",
        "//machina/c/eager:immediate_execution_tensor_handle",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "@com_google_absl//absl/types:optional",
    ],
)

cc_library(
    name = "tf_signature_def_function",
    srcs = [
        "tf_signature_def_function.cc",
    ],
    hdrs = [
        "tf_signature_def_function.h",
    ],
    deps = [
        ":flat_tensor_function",
        "//machina/c/eager:abstract_tensor_handle",
        "//machina/c/eager:immediate_execution_context",
        "//machina/c/eager:immediate_execution_operation",
        "//machina/c/eager:immediate_execution_tensor_handle",
        "//machina/c/experimental/saved_model/core:signature_def_function",
        "//machina/c/experimental/saved_model/core:signature_def_function_metadata",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "tf_signature_def_function_revival_state",
    hdrs = [
        "tf_signature_def_function_revival_state.h",
    ],
    deps = [
        ":tf_signature_def_function",
        "//machina/c/eager:immediate_execution_tensor_handle",
        "//machina/c/experimental/saved_model/core:signature_def_function_metadata",
        "//machina/core:protos_all_cc",
        "@com_google_absl//absl/types:optional",
    ],
)

load(
    "//machina:machina.bzl",
    "tf_copts",
    "tf_cuda_library",
)
load("//machina/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = [":friends"],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    packages = ["//machina/..."],
)

tf_cuda_library(
    name = "mlir_c_api",
    srcs = [
        "c_api_unified_experimental_mlir.cc",
    ],
    copts = tf_copts(),
    deps = [
        "//machina/c:c_api",
        "//machina/c:tensor_interface",
        "//machina/c:tf_status_helper",
        "//machina/c:tf_status_internal",
        "//machina/c/eager:abstract_context",
        "//machina/c/eager:abstract_operation",
        "//machina/c/eager:abstract_tensor_handle",
        "//machina/c/eager:c_api",
        "//machina/c/eager:c_api_internal",
        "//machina/c/eager:c_api_unified_internal",
        "//machina/compiler/mlir/machina",
        "//machina/compiler/mlir/machina:convert_type",
        "//machina/compiler/mlir/machina:dump_mlir_util",
        "//machina/compiler/mlir/machina:error_util",
        "//machina/compiler/mlir/machina:machina_types",
        "//machina/compiler/mlir/machina/transforms:machina_passes",
        "//machina/compiler/mlir/tf2xla/api/v2:tf_executor_to_graph",
        "//machina/core:framework",
        "//machina/core:protos_all_cc",
        "//machina/core/lib/llvm_rtti",
        "//machina/core/platform:errors",
        "//machina/core/platform:refcount",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:Support",
        "@local_xla//xla/tsl/platform:errors",
        "@local_xla//xla/tsl/platform:statusor",
    ],
)

cc_library(
    name = "mlir_c_api_registration",
    srcs = ["c_api_unified_experimental_mlir_registration.cc"],
    deps = [
        ":mlir_c_api",
        "//machina/c/eager:c_api_unified_internal",
    ],
    alwayslink = 1,
)

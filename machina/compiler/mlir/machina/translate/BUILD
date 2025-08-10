load("//machina/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "import_model",
    srcs = [
        "import_model.cc",
    ],
    hdrs = [
        "import_model.h",
    ],
    deps = [
        ":mlir_roundtrip_flags",
        "//machina/cc/saved_model:bundle_v2",
        "//machina/cc/saved_model:constants",
        "//machina/cc/saved_model:loader_lite",
        "//machina/cc/saved_model:loader_util",
        "//machina/compiler/mlir:op_or_arg_name_mapper",
        "//machina/compiler/mlir/machina",
        "//machina/compiler/mlir/machina:convert_tensor",
        "//machina/compiler/mlir/machina:convert_type",
        "//machina/compiler/mlir/machina:dump_mlir_util",
        "//machina/compiler/mlir/machina:error_util",
        "//machina/compiler/mlir/machina:mlir_import_options",
        "//machina/compiler/mlir/machina:translate_utils",
        "//machina/compiler/mlir/machina/transforms:initialize_variables_in_session_init",
        "//machina/compiler/mlir/machina/transforms:lift_variables_lib",
        "//machina/compiler/mlir/machina/transforms:mark_initialized_variables_lib",
        "//machina/compiler/mlir/machina/transforms:machina_passes",
        "//machina/compiler/mlir/machina/transforms:tf_saved_model_passes",
        "//machina/compiler/mlir/tf2xla/api/v2:graph_to_tf_executor",
        "//machina/compiler/tf2xla:functionalize_control_flow",
        "//machina/core:core_cpu_base",
        "//machina/core:framework",
        "//machina/core:framework_internal",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core/common_runtime:function_body",
        "//machina/core/platform:types",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:Support",
        "@local_xla//xla:status_macros",
    ],
)

cc_library(
    name = "mlir_roundtrip_flags",
    hdrs = ["mlir_roundtrip_flags.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//machina/compiler/mlir/tf2xla/api/v2:mlir_roundtrip_flags",
    ],
)

cc_library(
    name = "mlir_import_options",
    hdrs = ["mlir_import_options.h"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "translate_lib",
    srcs = ["tf_mlir_translate.cc"],
    hdrs = ["tf_mlir_translate.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":import_model",
        "//machina/cc/saved_model:bundle_v2",
        "//machina/cc/saved_model:loader_lite",
        "//machina/cc/saved_model:reader",
        "//machina/compiler/mlir/machina:mlir_import_options",
        "//machina/core:core_cpu",
        "//machina/core:core_cpu_base",
        "//machina/core:ops",
        "//machina/core:protos_all_cc",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@toolchain-project//mlir:IR",
    ],
)

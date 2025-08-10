load("//machina:machina.bzl", "tf_cc_test")
load(
    "//machina:machina.default.bzl",
    "get_compatible_with_portable",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = [
        # For TFLite Converter integration.
        "//machina/compiler/mlir/lite:__subpackages__",
        "//machina/compiler/mlir/quantization:__subpackages__",
        "//machina/compiler/mlir/machina_to_stablehlo:__subpackages__",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "types",
    hdrs = ["types.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [],
)

cc_library(
    name = "component",
    hdrs = ["component.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//machina/compiler/mlir/quantization/stablehlo:quantization_config_proto_cc",
        "@com_google_absl//absl/status:statusor",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Support",
    ],
)

cc_library(
    name = "config",
    srcs = ["config.cc"],
    hdrs = ["config.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//machina/compiler/mlir/quantization/stablehlo:quantization_config_proto_cc",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/strings:string_view",
    ],
)

tf_cc_test(
    name = "config_test",
    srcs = ["config_test.cc"],
    deps = [
        ":config",
        "//machina/compiler/mlir/quantization/stablehlo:quantization_config_proto_cc",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "io",
    srcs = ["io.cc"],
    hdrs = ["io.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@local_xla//xla/tsl/platform:env",
        "@local_xla//xla/tsl/platform:errors",
        "@local_xla//xla/tsl/platform:statusor",
    ],
)

tf_cc_test(
    name = "io_test",
    srcs = ["io_test.cc"],
    deps = [
        ":io",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@local_xla//xla/tsl/platform:env",
        "@local_xla//xla/tsl/platform:status",
        "@local_xla//xla/tsl/platform:status_matchers",
        "@local_xla//xla/tsl/platform:types",
    ],
)

cc_library(
    name = "graph_def",
    srcs = [],
    hdrs = ["graph_def.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//machina/core:protos_all_cc",
    ],
)

tf_cc_test(
    name = "graph_def_test",
    srcs = ["graph_def_test.cc"],
    deps = [
        ":graph_def",
        "//machina/core:protos_all_cc",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:protobuf",
    ],
)

cc_library(
    name = "debugger",
    srcs = ["debugger.cc"],
    hdrs = ["debugger.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//machina/compiler/mlir/quantization/machina:tf_quant_ops",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Support",
    ],
)

cc_library(
    name = "permutation",
    hdrs = ["permutation.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:Support",
    ],
)

tf_cc_test(
    name = "permutation_test",
    srcs = ["permutation_test.cc"],
    deps = [
        ":permutation",
        "@com_google_googletest//:gtest_main",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:Support",
    ],
)

cc_library(
    name = "saved_model_export",
    srcs = ["saved_model_export.cc"],
    hdrs = ["saved_model_export.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":io",
        ":pass_pipeline",
        ":saved_model_import",
        ":types",
        "//machina/compiler/mlir/quantization/stablehlo:quantization_config_proto_cc",
        "//machina/compiler/mlir/quantization/machina:exported_model_proto_cc",
        "//machina/compiler/mlir/quantization/machina:passes",
        "//machina/compiler/mlir/quantization/machina/cc:convert_asset_args",
        "//machina/compiler/mlir/quantization/machina/cc:run_passes",
        "//machina/compiler/mlir/quantization/machina/python:unfreeze_constants",
        "//machina/compiler/mlir/machina",
        "//machina/compiler/mlir/machina:mlir_roundtrip_flags",
        "//machina/compiler/mlir/machina/transforms:verify_no_outside_compilation_markers_pass",
        "//machina/compiler/mlir/tf2xla/api/v2:tf_executor_to_graph",
        "//machina/core:core_cpu_base",
        "//machina/core:framework",
        "//machina/core:protos_all_cc",
        "//machina/core/protobuf:for_core_protos_cc",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:Support",
        "@local_xla//xla/tsl/platform:errors",
        "@local_xla//xla/tsl/platform:statusor",
    ],
)

tf_cc_test(
    name = "saved_model_export_test",
    srcs = ["saved_model_export_test.cc"],
    deps = [
        ":saved_model_export",
        "//machina/compiler/mlir/quantization/common:test_base",
        "//machina/compiler/mlir/quantization/machina:exported_model_proto_cc",
        "//machina/core:all_kernels",  # buildcleaner: keep Required to export to GraphDef
        "//machina/core:core_cpu_base",
        "//machina/core:ops",  # buildcleaner: keep Required to export to GraphDef
        "//machina/core:protos_all_cc",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:IR",
        "@local_tsl//tsl/platform:protobuf",
        "@local_xla//xla/tsl/platform:status_matchers",
        "@local_xla//xla/tsl/platform:statusor",
    ],
)

cc_library(
    name = "saved_model_import",
    srcs = ["saved_model_import.cc"],
    hdrs = ["saved_model_import.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":types",
        "//machina/cc/saved_model:loader",
        "//machina/cc/saved_model:reader",
        "//machina/compiler/mlir/quantization/stablehlo:quantization_config_proto_cc",
        "//machina/compiler/mlir/quantization/machina:quantize_preprocess",
        "//machina/compiler/mlir/machina:mlir_import_options",
        "//machina/compiler/mlir/machina:translate_lib",
        "//machina/core/protobuf:for_core_protos_cc",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@local_xla//xla/tsl/platform:errors",
        "@local_xla//xla/tsl/platform:statusor",
    ],
)

tf_cc_test(
    name = "saved_model_import_test",
    srcs = ["saved_model_import_test.cc"],
    deps = [
        ":saved_model_import",
        ":types",
        "//machina/compiler/mlir/quantization/common:test_base",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_googletest//:gtest_main",
        "@toolchain-project//mlir:IR",
    ],
)

cc_library(
    name = "pass_pipeline",
    srcs = ["pass_pipeline.cc"],
    hdrs = ["pass_pipeline.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//machina/compiler/mlir/quantization/stablehlo:bridge_passes",
        "//machina/compiler/mlir/quantization/stablehlo:passes",
        "//machina/compiler/mlir/quantization/stablehlo:quantization_config_proto_cc",
        "//machina/compiler/mlir/quantization/machina:passes",
        "//machina/compiler/mlir/quantization/machina:quantization_options_proto_cc",
        "//machina/compiler/mlir/machina/transforms:machina_passes",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:ReconcileUnrealizedCasts",
        "@toolchain-project//mlir:Transforms",
        "@local_xla//xla/mlir_hlo:mhlo_passes",
        "@stablehlo//:stablehlo_passes",
    ],
)

cc_library(
    name = "pre_calibration",
    srcs = ["pre_calibration.cc"],
    hdrs = ["pre_calibration.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = [
        "//machina/compiler/mlir/quantization/stablehlo:__subpackages__",
        "//machina/compiler/mlir/quantization/machina:__subpackages__",
    ],
    deps = [
        ":component",
        ":pass_pipeline",
        "//machina/compiler/mlir/quantization/stablehlo:quantization_config_proto_cc",
        "//machina/compiler/mlir/quantization/machina:passes",
        "//machina/compiler/mlir/quantization/machina:quantization_options_proto_cc",
        "//machina/compiler/mlir/quantization/machina/cc:run_passes",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/log:die_if_null",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Pass",
        "@local_xla//xla/tsl/platform:errors",
    ],
)

tf_cc_test(
    name = "pre_calibration_test",
    srcs = ["pre_calibration_test.cc"],
    deps = [
        ":config",
        ":pre_calibration",
        "//machina/compiler/mlir/quantization/common:test_base",
        "//machina/compiler/mlir/quantization/stablehlo:quantization_config_proto_cc",
        "//machina/compiler/mlir/quantization/machina:quantization_options_proto_cc",
        "//machina/compiler/mlir/quantization/machina:tf_quant_ops",
        "//machina/compiler/mlir/machina",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest_main",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Support",
        "@local_xla//xla/tsl/platform:status_matchers",
    ],
)

cc_library(
    name = "report",
    srcs = ["report.cc"],
    hdrs = ["report.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":io",
        "//machina/compiler/mlir/quantization/common:lift_as_function_call",
        "//machina/compiler/mlir/quantization/stablehlo:quantization_config_proto_cc",
        "//machina/compiler/mlir/machina",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Support",
        "@local_tsl//tsl/platform:protobuf",
    ],
)

tf_cc_test(
    name = "report_test",
    srcs = ["report_test.cc"],
    deps = [
        ":io",
        ":report",
        "//machina/compiler/mlir/quantization/common:test_base",
        "//machina/compiler/mlir/quantization/stablehlo:quantization_config_proto_cc",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
        "@toolchain-project//mlir:IR",
        "@local_tsl//tsl/platform:protobuf",
        "@local_xla//xla/tsl/platform:status_matchers",
    ],
)

cc_library(
    name = "context",
    srcs = [],
    hdrs = ["context.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = ["//visibility:public"],
    deps = [
        "@toolchain-project//mlir:FuncExtensions",
        "@toolchain-project//mlir:IR",
    ],
)

cc_library(
    name = "post_calibration",
    srcs = ["post_calibration.cc"],
    hdrs = ["post_calibration.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":component",
        ":config",
        ":pass_pipeline",
        "//machina/compiler/mlir/quantization/stablehlo:quantization_config_proto_cc",
        "//machina/compiler/mlir/quantization/stablehlo/instrumentations:save_report",
        "//machina/compiler/mlir/quantization/machina:passes",
        "//machina/compiler/mlir/quantization/machina/cc:run_passes",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/log:die_if_null",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Pass",
        "@local_xla//xla/mlir_hlo:mhlo_passes",
        "@local_xla//xla/tsl/platform:errors",
    ],
)

cc_library(
    name = "static_range_ptq",
    srcs = ["static_range_ptq.cc"],
    hdrs = ["static_range_ptq.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":component",
        ":context",
        ":post_calibration",
        ":pre_calibration",
        ":saved_model_export",
        ":saved_model_import",
        ":types",
        "//machina/compiler/mlir/quantization/stablehlo:quantization_config_proto_cc",
        "//machina/compiler/mlir/quantization/stablehlo/cc/calibration:component",
        "//machina/compiler/mlir/quantization/machina:exported_model_proto_cc",
        "//machina/compiler/mlir/quantization/machina:quantization_options_proto_cc",
        "//machina/compiler/mlir/quantization/machina/python:py_function_lib",
        "//machina/core:core_cpu_base",
        "//machina/core:protos_all_cc",
        "//machina/core/protobuf:for_core_protos_cc",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Pass",
        "@local_xla//xla/tsl/platform:statusor",
    ],
)

cc_library(
    name = "weight_only_ptq",
    srcs = ["weight_only_ptq.cc"],
    hdrs = ["weight_only_ptq.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":component",
        ":config",
        ":context",
        ":pass_pipeline",
        ":saved_model_export",
        ":saved_model_import",
        ":types",
        "//machina/compiler/mlir/quantization/stablehlo:quantization_config_proto_cc",
        "//machina/compiler/mlir/quantization/stablehlo/instrumentations:save_report",
        "//machina/compiler/mlir/quantization/machina:exported_model_proto_cc",
        "//machina/compiler/mlir/quantization/machina:passes",
        "//machina/compiler/mlir/quantization/machina/cc:run_passes",
        "//machina/compiler/mlir/quantization/machina/python:py_function_lib",
        "//machina/core:core_cpu_base",
        "//machina/core/protobuf:for_core_protos_cc",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:die_if_null",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Pass",
        "@local_xla//xla/mlir_hlo:mhlo_passes",
        "@local_xla//xla/tsl/platform:errors",
        "@local_xla//xla/tsl/platform:statusor",
    ],
)

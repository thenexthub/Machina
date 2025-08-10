load("@rules_cc//cc:cc_library.bzl", "cc_library")
load(
    "//machina:machina.bzl",
    "tf_cc_test",
)
load(
    "//machina:machina.default.bzl",
    "get_compatible_with_portable",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    # By default, these targets should only be used within the quantization library.
    default_visibility = [
        "//learning/brain/mlir/quantization:__subpackages__",
        "//machina/compiler/mlir/quantization:__subpackages__",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "save_variables",
    srcs = ["save_variables.cc"],
    hdrs = ["save_variables.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//machina/compiler/mlir/machina",
        "//machina/core:framework",
        "//machina/core/ir/importexport:convert_tensor",
        "//machina/core/util/tensor_bundle",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@local_xla//xla/tsl/platform:env",
        "@local_xla//xla/tsl/platform:logging",
        "@local_xla//xla/tsl/platform:status",
    ],
)

tf_cc_test(
    name = "save_variables_test",
    srcs = ["save_variables_test.cc"],
    deps = [
        ":save_variables",
        "//machina/compiler/mlir/machina",
        "//machina/compiler/mlir/machina:machina_types",
        "//machina/core:framework",
        "//machina/core:protos_all_cc",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core/framework:tensor_testutil",
        "//machina/core/util/tensor_bundle",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_googletest//:gtest_main",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Parser",
        "@local_xla//xla/tsl/platform:status",
        "@local_xla//xla/tsl/platform:status_matchers",
    ],
)

cc_library(
    name = "const_op_size",
    srcs = ["const_op_size.cc"],
    hdrs = ["const_op_size.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//machina/compiler/mlir/machina",
        "//machina/compiler/mlir/machina:machina_remaining_ops",
        "//machina/compiler/mlir/machina:machina_types",
        "@com_google_absl//absl/algorithm:container",
        "@toolchain-project//mlir:IR",
    ],
)

tf_cc_test(
    name = "const_op_size_test",
    srcs = ["const_op_size_test.cc"],
    deps = [
        ":const_op_size",
        "//machina/compiler/mlir/machina",
        "//machina/core:test",
        "//machina/core:test_main",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Parser",
        "@toolchain-project//mlir:Support",
    ],
)

cc_library(
    name = "convert_asset_args",
    srcs = ["convert_asset_args.cc"],
    hdrs = ["convert_asset_args.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//machina/compiler/mlir/quantization/common:func",
        "//machina/compiler/mlir/machina",
        "//machina/compiler/mlir/machina:import_model",
        "//machina/core/protobuf:for_core_protos_cc",
        "@com_google_absl//absl/algorithm:container",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Support",
    ],
)

tf_cc_test(
    name = "convert_asset_args_test",
    srcs = ["convert_asset_args_test.cc"],
    deps = [
        ":convert_asset_args",
        "//machina/compiler/mlir/machina",
        "//machina/compiler/mlir/machina:machina_types",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core/protobuf:for_core_protos_cc",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Parser",
        "@toolchain-project//mlir:Support",
    ],
)

cc_library(
    name = "run_passes",
    srcs = ["run_passes.cc"],
    hdrs = ["run_passes.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//machina/compiler/mlir/quantization/machina/debugging:mlir_dump",
        "//machina/compiler/mlir/machina:error_util",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Pass",
        "@local_xla//xla/tsl/platform:errors",
        "@local_xla//xla/tsl/platform:statusor",
    ],
)

cc_library(
    name = "constant_fold",
    srcs = [
        "constant_fold.cc",
    ],
    hdrs = [
        "constant_fold.h",
    ],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//machina/compiler/mlir/quantization/common:lift_as_function_call",
        "//machina/compiler/mlir/machina",
        "//machina/compiler/mlir/machina/transforms:constant_fold_utils",
        "@com_google_absl//absl/container:flat_hash_set",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Support",
        "@toolchain-project//mlir:TransformUtils",
    ],
)

tf_cc_test(
    name = "constant_fold_test",
    srcs = ["constant_fold_test.cc"],
    deps = [
        ":constant_fold",
        "//machina/compiler/mlir/quantization/common:attrs_and_constraints",
        "//machina/compiler/mlir/quantization/common:test_base",
        "//machina/compiler/mlir/machina",
        "//machina/core:machina",
        "//machina/core:test",
        "//machina/core:test_main",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Support",
        "@toolchain-project//mlir:TransformUtils",
    ],
)

cc_library(
    name = "quantization_unit_loc",
    srcs = ["quantization_unit_loc.cc"],
    hdrs = ["quantization_unit_loc.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//machina/compiler/mlir/quantization/machina:quantization_options_proto_cc",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Support",
    ],
)

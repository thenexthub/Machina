load("@toolchain-project//mlir:tblgen.bzl", "gentbl_cc_library")
load("@local_xla//xla/tsl:tsl.default.bzl", "tsl_pybind_extension")
load("@local_xla//xla/tsl/platform:build_config_root.bzl", "if_static")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//machina:pytype.default.bzl", "pytype_strict_library")
load("//machina:strict.default.bzl", "py_strict_test")
load("//machina:machina.bzl", "tf_cc_test")
load("//machina:machina.default.bzl", "get_compatible_with_portable")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = [
        ":friends",
        "//machina/tools/pip_package:__subpackages__",
    ],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    packages = [
        "//platforms/darwinn/tools/visualization/graph_conversions/...",
        "//machina/compiler/mlir/lite/...",
        "//machina/compiler/mlir/quantization/...",
        "//machina/compiler/mlir/quantization/machina/...",
        "//machina/compiler/tests/...",
    ],
)

tsl_pybind_extension(
    name = "stablehlo_extension",
    srcs = [
        "stablehlo.cc",
        "@stablehlo//:stablehlo/integrations/python/StablehloApi.cpp",
    ],
    hdrs = [
        "@stablehlo//:stablehlo/integrations/python/StablehloApi.h",
    ],
    copts = [
        "-fexceptions",
        "-frtti",
    ],
    features = ["-use_header_modules"],
    deps = [
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:CAPIIR",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:MLIRBindingsPythonHeadersAndDeps",
        "@local_xla//third_party/python_runtime:headers",
        "@nanobind",
        "@stablehlo//:stablehlo_capi",
    ],
)

pytype_strict_library(
    name = "stablehlo",
    srcs = ["stablehlo.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":stablehlo_extension",
    ],
)

py_strict_test(
    name = "stablehlo_test",
    srcs = ["stablehlo_test.py"],
    deps = [
        ":stablehlo",
        #internal proto upb dep
    ],
)

gentbl_cc_library(
    name = "legalize_tf_patterns_inc_gen",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = [
        (
            ["-gen-rewriters"],
            "transforms/generated_legalize_tf.inc",
        ),
    ],
    tblgen = "@toolchain-project//mlir:mlir-tblgen",
    td_file = "transforms/legalize_tf_patterns.td",
    deps = [
        "//machina/compiler/mlir/machina:machina_ops_td_files",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:FuncTdFiles",
        "@toolchain-project//mlir:TensorOpsTdFiles",
        "@local_xla//xla/mlir_hlo:hlo_ops_td_files",
    ],
)

cc_library(
    name = "fold_broadcast_pass",
    srcs = [
        "transforms/fold_broadcast_pass.cc",
    ],
    hdrs = [
        "transforms/fold_broadcast_pass.h",
    ],
    compatible_with = get_compatible_with_portable(),
    copts = [
        "-Ithird_party",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:Support",
        "@toolchain-project//mlir:TransformUtils",
        "@toolchain-project//mlir:Transforms",
        "@local_xla//xla/mlir_hlo",
    ],
    alwayslink = 1,
)

cc_library(
    name = "legalize_utils",
    srcs = ["transforms/utils.cc"],
    hdrs = ["transforms/utils.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:IR",
        "@local_xla//xla/mlir_hlo",
    ],
)

tf_cc_test(
    name = "legalize_utils_test",
    srcs = ["transforms/utils_test.cc"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":legalize_utils",
        "@com_google_googletest//:gtest_main",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Support",
        "@local_xla//xla/mlir_hlo",
    ],
)

cc_library(
    name = "legalize_tf",
    srcs = [
        "transforms/generated_legalize_tf.inc",
        "transforms/legalize_tf.cc",
    ],
    hdrs = [
        "transforms/legalize_tf_passes.h",
    ],
    compatible_with = get_compatible_with_portable(),
    visibility = ["//visibility:public"],
    deps = [
        ":legalize_tf_patterns_inc_gen",
        ":legalize_utils",
        "//machina/compiler/mlir/machina",
        "//machina/compiler/mlir/machina:dynamic_shape_utils",
        "//machina/compiler/mlir/machina:xla_sharding_util",
        "//machina/core:framework",
        "//machina/core/kernels:conv_grad_shape_utils",
        "@com_google_absl//absl/status",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:ArithDialect",
        "@toolchain-project//mlir:Dialect",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:MemRefDialect",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:ShapeDialect",
        "@toolchain-project//mlir:Support",
        "@toolchain-project//mlir:TensorDialect",
        "@toolchain-project//mlir:TransformUtils",
        "@local_tsl//tsl/platform:bfloat16",
        "@local_tsl//tsl/platform:tensor_float_32_hdr_lib",
        "@local_xla//xla:xla_data_proto_cc",
        "@local_xla//xla/hlo/builder:padding",
        "@local_xla//xla/hlo/builder:sharding_builder",
        "@local_xla//xla/hlo/builder/lib:conv_grad_size_util",
        "@local_xla//xla/hlo/translate/hlo_to_mhlo:attribute_importer",
        "@local_xla//xla/mlir_hlo",
        "@local_xla//xla/mlir_hlo:convert_op_folder",
        "@local_xla//xla/tsl/platform:status",
        "@stablehlo//:chlo_ops",
        "@stablehlo//:stablehlo_ops",
    ] + if_static(["@local_tsl//tsl/platform:tensor_float_32_utils"]),
)

cc_library(
    name = "tf_stablehlo",
    srcs = [
        "transforms/tf_stablehlo_pass.cc",
    ],
    hdrs = [
        "transforms/tf_stablehlo_pass.h",
    ],
    compatible_with = get_compatible_with_portable(),
    copts = [
        "-Ithird_party",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":legalize_tf",
        "//machina/compiler/mlir/machina",
        "//machina/compiler/mlir/machina/transforms:lower_tf_lib",
        "//machina/compiler/mlir/tf2xla/transforms:xla_legalize_tf_with_tf2xla",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:ArithDialect",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:ShapeDialect",
        "@toolchain-project//mlir:TensorDialect",
        "@toolchain-project//mlir:TransformUtils",
        "@toolchain-project//mlir:Transforms",
        "@local_xla//xla/mlir_hlo",
        "@local_xla//xla/mlir_hlo:hlo_dialect_registration",
        "@local_xla//xla/mlir_hlo:mhlo_passes",
        "@local_xla//xla/mlir_hlo:type_conversion",
        "@stablehlo//:chlo_ops",
        "@stablehlo//:register",
    ],
    alwayslink = 1,
)

# LINT.IfChange(legalize_tf_xla_call_module_to_stablehlo_pass)
cc_library(
    name = "legalize_tf_xla_call_module_to_stablehlo_pass",
    srcs = [
        "transforms/legalize_tf_xla_call_module_to_stablehlo_pass.cc",
    ],
    hdrs = [
        "transforms/legalize_tf_xla_call_module_to_stablehlo_pass.h",
    ],
    compatible_with = get_compatible_with_portable(),
    copts = [
        "-Ithird_party",
    ],
    deps = [
        "//machina/compiler/mlir/machina",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:QuantOps",
        "@toolchain-project//mlir:ShapeDialect",
        "@toolchain-project//mlir:Support",
        "@toolchain-project//mlir:TransformUtils",
        "@toolchain-project//mlir:Transforms",
        "@stablehlo//:stablehlo_ops",
        "@stablehlo//:stablehlo_serialization",
        "@stablehlo//:vhlo_ops",
    ],
    alwayslink = 1,
)
# LINT.ThenChange(//machina/compiler/mlir/lite/stablehlo:legalize_tf_xla_call_module_to_stablehlo_pass)

cc_library(
    name = "fuse_convolution_pass",
    srcs = [
        "transforms/mhlo_passes/fuse_convolution_pass.cc",
    ],
    hdrs = [
        "transforms/mhlo_passes/fuse_convolution_pass.h",
    ],
    compatible_with = get_compatible_with_portable(),
    copts = [
        "-Ithird_party",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//machina/compiler/mlir/quantization/common:attrs_and_constraints",
        "//machina/compiler/mlir/utils:validators",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:Dialect",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:ShapeDialect",
        "@toolchain-project//mlir:Support",
        "@toolchain-project//mlir:TransformUtils",
        "@toolchain-project//mlir:Transforms",
        "@local_xla//xla/mlir_hlo",
    ],
    alwayslink = 1,
)

cc_library(
    name = "unfuse_batch_norm_pass",
    srcs = [
        "transforms/mhlo_passes/unfuse_batch_norm_pass.cc",
    ],
    hdrs = [
        "transforms/mhlo_passes/unfuse_batch_norm_pass.h",
    ],
    compatible_with = get_compatible_with_portable(),
    copts = [
        "-Ithird_party",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//machina/compiler/mlir/machina:machina_ops",
        "//machina/compiler/mlir/machina:machina_types",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:ShapeDialect",
        "@toolchain-project//mlir:Support",
        "@toolchain-project//mlir:TransformUtils",
        "@local_xla//xla/mlir_hlo",
    ],
    alwayslink = 1,
)

cc_library(
    name = "rename_entrypoint_to_main",
    srcs = [
        "transforms/rename_entrypoint_to_main.cc",
    ],
    hdrs = [
        "transforms/rename_entrypoint_to_main.h",
    ],
    compatible_with = get_compatible_with_portable(),
    copts = [
        "-Ithird_party",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//machina/compiler/mlir/machina",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:Support",
    ],
    alwayslink = 1,
)

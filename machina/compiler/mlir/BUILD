# Description:
#   TensorFlow/TensorFlow Lite/XLA MLIR dialects and tools.

load(
    "//machina:machina.bzl",
    "tf_cc_binary",
    "tf_cc_test",
)
load("//machina:machina.default.bzl", "filegroup")
load("//machina/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

package_group(
    name = "subpackages",
    packages = ["//machina/compiler/mlir/..."],
)

exports_files(glob(["g3doc/*.md"] + ["g3doc/_includes/tf_passes.md"]))

# To reference all tablegen files here when checking for updates to them.
filegroup(
    name = "td_files",
    srcs = glob(["**/*.td"]),
)

cc_library(
    name = "op_or_arg_name_mapper",
    srcs = ["op_or_arg_name_mapper.cc"],
    hdrs = ["op_or_arg_name_mapper.h"],
    deps = [
        "//machina/compiler/mlir/utils:name_utils",
        "@com_google_absl//absl/strings",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Support",
    ],
)

tf_cc_test(
    name = "op_or_arg_name_mapper_test",
    srcs = ["op_or_arg_name_mapper_test.cc"],
    deps = [
        ":op_or_arg_name_mapper",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Support",
    ],
)

cc_library(
    name = "tf_mlir_opt_main",
    testonly = True,
    srcs = ["tf_mlir_opt_main.cc"],
    deps = [
        ":init_mlir",
        ":passes",
        ":register_common_dialects",
        "//machina/compiler/mlir/quantization/stablehlo:bridge_passes",
        "//machina/compiler/mlir/machina:mlprogram_util",
        "//machina/compiler/mlir/machina/transforms:machina_passes",
        "//machina/compiler/mlir/machina/transforms:machina_test_passes",
        "//machina/compiler/mlir/machina/transforms:tf_graph_optimization_pass",
        "//machina/compiler/mlir/machina/transforms:tf_saved_model_passes",  # buildcleaner:keep
        "//machina/compiler/mlir/machina/transforms/host_runtime:lower_cluster_to_runtime_ops",
        "//machina/compiler/mlir/machina/transforms/host_runtime:runtime_passes",
        "//machina/compiler/mlir/machina/transforms/sparsecore:sparsecore_passes",
        "//machina/compiler/mlir/tf2xla:compile_mlir_util",
        "//machina/compiler/mlir/tf2xla/internal/passes:clustering_passes",
        "//machina/compiler/mlir/tf2xla/internal/passes:mlir_to_graph_passes",
        "//machina/compiler/mlir/tf2xla/transforms:tf_xla_passes",
        "//machina/compiler/mlir/tf2xla/transforms:xla_legalize_tf",
        "@toolchain-project//mlir:AllPassesAndDialects",
        "@toolchain-project//mlir:MlirOptLib",
        "@toolchain-project//mlir:RegisterAllDialects",
        "@toolchain-project//mlir:RegisterAllPasses",
        "@toolchain-project//mlir:Support",
        "@toolchain-project//mlir:Transforms",
        "@local_xla//xla/mlir/framework/transforms:passes",
        "@local_xla//xla/mlir_hlo:all_passes",
    ],
)

cc_library(
    name = "passes",
    visibility = ["//visibility:public"],
    deps = [
        "@toolchain-project//mlir:AffineDialect",
        "@toolchain-project//mlir:QuantOps",
        # Link jit lib to link JIT devices required to run
        # xla-legalize-tf-with-tf2xla pass.
        "//machina/compiler/jit",
        "//machina/compiler/mlir/machina",
        "//machina/compiler/mlir/machina/transforms:machina_passes",
        "//machina/compiler/mlir/machina/transforms:tf_dialect_passes",
        "//machina/compiler/mlir/tf2xla:compile_mlir_util",
    ],
)

cc_library(
    name = "init_mlir",
    srcs = ["init_mlir.cc"],
    hdrs = ["init_mlir.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//machina/core:lib",
        "@toolchain-project//toolchain:Support",
    ],
)

cc_library(
    name = "mlir_graph_optimization_pass",
    srcs = ["mlir_graph_optimization_pass.cc"],
    hdrs = ["mlir_graph_optimization_pass.h"],
    deps = [
        "//machina/compiler/mlir/machina",
        "//machina/compiler/mlir/machina:attribute_utils",
        "//machina/compiler/mlir/machina:device_util",
        "//machina/compiler/mlir/machina:dump_mlir_util",
        "//machina/compiler/mlir/machina:mlir_roundtrip_flags",
        "//machina/compiler/mlir/tf2xla:mlir_bridge_rollout_policy",
        "//machina/compiler/mlir/tf2xla/api/v2:graph_to_tf_executor",
        "//machina/compiler/mlir/tf2xla/api/v2:tf_executor_to_graph",
        "//machina/core:core_cpu",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core/common_runtime:device_set",
        "//machina/core/protobuf:for_core_protos_cc",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:string_view",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:ArithDialect",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:FuncExtensions",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:ShapeDialect",
    ],
    alwayslink = 1,
)

cc_library(
    name = "mlir_graph_optimization_pass_registration",
    srcs = [
        "mlir_graph_optimization_pass_registration.cc",
    ],
    deps = [
        ":mlir_graph_optimization_pass",
        "//machina/core:core_cpu",
    ],
    alwayslink = 1,
)

# This should just be a wrapper around tf_mlir_opt_main. Don't add
# direct dependencies to this binary.
tf_cc_binary(
    name = "tf-opt",
    testonly = True,
    deps = [
        ":tf_mlir_opt_main",
    ],
)

tf_cc_binary(
    name = "tf-reduce",
    testonly = True,
    srcs = ["tf_mlir_reduce_main.cc"],
    deps = [
        ":init_mlir",
        ":register_common_dialects",
        "//machina/compiler/mlir/machina",
        "//machina/compiler/mlir/machina/transforms:machina_reduce_patterns_inc_gen",
        "@toolchain-project//mlir:AllPassesAndDialects",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:MlirReduceLib",
        "@toolchain-project//mlir:Reducer",
    ],
)

cc_library(
    name = "register_common_dialects",
    testonly = True,
    srcs = ["register_common_dialects.cc"],
    hdrs = ["register_common_dialects.h"],
    deps = [
        "//machina/compiler/mlir/machina",
        "//machina/compiler/mlir/machina:mlprogram_util",
        "//machina/compiler/mlir/tools/kernel_gen/ir:tf_framework_ops",
        "//machina/core/ir/types:Dialect",
        "@toolchain-project//mlir:AllExtensions",
        "@toolchain-project//mlir:AllPassesAndDialects",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:MlirOptLib",
        "@toolchain-project//mlir:QuantOps",
        "@toolchain-project//mlir:RegisterAllDialects",
        "@toolchain-project//mlir:RegisterAllExtensions",
        "@toolchain-project//mlir:RegisterAllPasses",
        "@toolchain-project//mlir:ShapeDialect",
        "@toolchain-project//mlir:TensorDialect",
        "@toolchain-project//mlir:TosaDialect",
        "@toolchain-project//mlir:Transforms",
        "@local_xla//xla/mlir/framework/ir:xla_framework",
        "@local_xla//xla/mlir_hlo:hlo_dialect_registration",
        "@stablehlo//:register",
    ],
)

tf_cc_test(
    name = "register_common_dialects_test",
    srcs = ["register_common_dialects_test.cc"],
    deps = [
        ":register_common_dialects",
        "@com_google_googletest//:gtest_main",
        "@toolchain-project//mlir:IR",
    ],
)

tf_cc_binary(
    name = "tf-mlir-translate",
    testonly = True,
    srcs = ["tf_mlir_translate_main.cc"],
    deps = [
        ":init_mlir",
        "//machina/compiler/mlir/machina:tf_xla_mlir_translate",
        "//machina/compiler/mlir/machina:translate_lib",
        "//machina/compiler/mlir/tf2xla/tests/registration:graph_to_tf_executor_registration",
        "//machina/compiler/mlir/tools:translate_cl_options",
        "//machina/compiler/mlir/tools:translate_registration",
        "//machina/core:lib",
        "//machina/core:machina",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Support",
        "@toolchain-project//mlir:TranslateLib",
        "@local_xla//xla/hlo/translate/hlo_to_mhlo:translate_registration",
        "@local_xla//xla/hlo/translate/mhlo_to_hlo:translate_registration",
    ],
)

tf_cc_test(
    name = "mlir_graph_optimization_pass_test",
    srcs = ["mlir_graph_optimization_pass_test.cc"],
    deps = [
        ":mlir_graph_optimization_pass",
        "//machina/core:core_cpu",
        "//machina/core:framework",
        "//machina/core:ops",
        "//machina/core:portable_gif_internal",
        "//machina/core:protos_all_cc",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core/common_runtime:device_set",
        "//machina/core/common_runtime:optimization_registry",
        "//machina/core/framework:tensor_testutil",
        "//machina/core/lib/monitoring:cell_reader",
        "//machina/core/platform:status",
        "//machina/core/protobuf:for_core_protos_cc",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:IR",
    ],
)

filegroup(
    name = "litfiles",
    srcs = glob(["runlit*py"]),
    visibility = ["//machina:__subpackages__"],
)

exports_files(["run_lit.sh"])

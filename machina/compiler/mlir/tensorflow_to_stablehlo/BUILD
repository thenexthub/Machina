load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//machina:machina.bzl", "tf_cc_binary")
load("//machina:machina.default.bzl", "filegroup", "get_compatible_with_portable")
load("//machina/compiler/mlir:glob_lit_test.bzl", "glob_lit_tests")
load("//machina/compiler/mlir/quantization/stablehlo:internal_visibility_allowlist.bzl", "internal_visibility_allowlist")

package_group(
    name = "internal_visibility_allowlist_package",
    packages = [
        "//machina/compiler/mlir/lite/...",
        "//machina/compiler/mlir/quantization/...",
        "//machina/compiler/mlir/machina_to_stablehlo/...",
        "//machina/compiler/mlir/tf2xla/transforms/...",
        "//machina/lite/...",
    ] + internal_visibility_allowlist(),
)

package(
    # copybara:uncomment default_applicable_licenses = ["@stablehlo//:license"],
    default_visibility = [
        ":internal_visibility_allowlist_package",
        "//machina:__pkg__",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "tf_to_stablehlo",
    srcs = [
        "tf_to_stablehlo.cc",
    ],
    hdrs = [
        "tf_to_stablehlo.h",
    ],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//machina/compiler/mlir/quantization/stablehlo/cc:saved_model_import",
        "//machina/compiler/mlir/quantization/machina:quantize_preprocess",
        "//machina/compiler/mlir/machina/transforms:shape_inference_pass",
        "//machina/core:core_cpu_base",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Parser",
        "@toolchain-project//mlir:Support",
        "@toolchain-project//mlir:Transforms",
        "@local_xla//xla/tsl/platform:errors",
        "@local_xla//xla/tsl/platform:statusor",
    ],
    alwayslink = True,
)

tf_cc_binary(
    name = "tf-to-stablehlo-translate",
    srcs = [
        "tf_to_stablehlo_translate.cc",
    ],
    visibility = [":internal_visibility_allowlist_package"],
    deps = [
        ":tf_to_stablehlo",
        "//machina/compiler/mlir:init_mlir",
        "//machina/compiler/mlir/machina",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:AllPassesAndDialects",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Parser",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:Support",
    ],
)

glob_lit_tests(
    name = "all_tests",
    data = [":test_utilities"],
    default_tags = [
        "no_oss",
        "no_pip",
    ],
    driver = "//machina/compiler/mlir/machina_to_stablehlo:run_lit.sh",
    size_override = {
    },
    tags_override = {
    },
    test_file_exts = ["mlir"],
)

# Bundle together all of the test utilities that are used by tests.
filegroup(
    name = "test_utilities",
    testonly = True,
    data = [
        ":tf-to-stablehlo-translate",
        "@toolchain-project//toolchain:FileCheck",
        "@toolchain-project//toolchain:not",
        "@toolchain-project//mlir:run_lit.sh",
    ],
)

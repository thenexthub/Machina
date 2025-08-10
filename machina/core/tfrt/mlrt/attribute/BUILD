load("@rules_cc//cc:cc_library.bzl", "cc_library")
load(
    "//machina:machina.bzl",
    "tf_cc_test",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = [
        # copybara:uncomment "//learning/brain/experimental/tfrt:__subpackages__",
        # copybara:uncomment "//smartass/brain/ops/tfrt_kernels:__subpackages__",
        "//machina/compiler/mlir/tfrt/transforms/mlrt:__subpackages__",
        "//machina/compiler/mlir/tfrt/translate/mlrt:__subpackages__",
    ],
)

cc_library(
    name = "attribute",
    srcs = ["attribute.cc"],
    hdrs = ["attribute.h"],
    deps = [
        "//machina/compiler/mlir/machina:convert_type",
        "//machina/compiler/mlir/machina:machina_attributes",
        "//machina/compiler/mlir/tfrt/translate/mlrt:mlir_to_bytecode",
        "//machina/core:protos_all_cc",
        "//machina/core/tfrt/mlrt/bytecode",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Support",
        "@local_xla//xla/tsl/platform:errors",
    ],
)

tf_cc_test(
    name = "attribute_test",
    srcs = ["attribute_test.cc"],
    deps = [
        ":attribute",
        "//machina/compiler/mlir/machina",  # build_cleaner: keep
        "//machina/compiler/mlir/machina:machina_attributes",
        "//machina/compiler/mlir/machina:machina_types",
        "//machina/compiler/mlir/tfrt/translate/mlrt:mlir_to_bytecode",
        "//machina/core:protos_all_cc",
        "//machina/core/tfrt/mlrt/bytecode",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:IR",
        "@local_xla//xla/tsl/platform:status_matchers",
        "@local_xla//xla/tsl/platform:statusor",
    ],
)

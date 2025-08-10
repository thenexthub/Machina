load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//machina:machina.bzl", "tf_cc_shared_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = [":friends"],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    packages = [
        # Authorized users go here.
        "//machina/core/tfrt/saved_model/...",
        "//machina/compiler/mlir/machina/...",
        "//learning/brain/tfrt/cpp_tests/gpu_inference/...",
        "//machina/compiler/mlir/tfrt/...",
        "//machina/compiler/mlir/tfrt/translate/...",
    ],
)

cc_library(
    name = "serialize_utils",
    srcs = ["serialize_utils.cc"],
    hdrs = ["serialize_utils.h"],
    deps = [
        "//machina/compiler/mlir/machina:dump_mlir_util",
        "//machina/core/platform:status",
        "//machina/core/tfrt/mlrt/bytecode",
        "//machina/core/tfrt/mlrt/bytecode:executable",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:Support",
        "@local_tsl//tsl/platform:env",
        "@tf_runtime//:bef",
    ],
)

tf_cc_shared_test(
    name = "serialize_utils_test",
    srcs = ["serialize_utils_test.cc"],
    data = [
        "//machina/compiler/mlir/tfrt/tests/saved_model:testdata",
    ],
    tags = ["no_oss"],
    deps = [
        ":serialize_utils",
        "//machina/compiler/mlir/machina",
        "//machina/compiler/mlir/tfrt:import_model",
        "//machina/compiler/mlir/tfrt/transforms/mlrt:import_model",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core/platform:path",
        "//machina/core/platform:resource_loader",
        "//machina/core/tfrt/fallback:fallback_state",
        "//machina/core/tfrt/mlrt/bytecode",
        "//machina/core/tfrt/saved_model:saved_model_testutil",
        "//machina/core/tfrt/saved_model:saved_model_util",
        "//machina/core/tfrt/utils",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Parser",
        "@local_tsl//tsl/platform:env",
        "@local_xla//xla/tsl/lib/core:status_test_util",
        "@tf_runtime//:bef",
    ],
)

load("//machina:machina.bzl", "tf_cc_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    licenses = ["notice"],
)

tf_cc_test(
    name = "saved_model_test",
    srcs = ["saved_model_test.cc"],
    data = [
        "testdata/test.mlir",
        "testdata/xla_launch.mlir",
        "testdata/xla_launch_xla_reduce_window.mlir",
    ],
    tags = ["no_oss"],
    deps = [
        "//machina/compiler/mlir/machina",
        "//machina/compiler/mlir/tfrt:import_model",
        "//machina/compiler/mlir/tfrt:saved_model",
        "//machina/compiler/mlir/tfrt:tfrt_compile_options",
        "//machina/core:core_cpu",
        "//machina/core:framework",
        "//machina/core:protos_all_cc",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core/platform:resource_loader",
        "//machina/core/tfrt/fallback:fallback_state",
        "//machina/core/tfrt/graph_executor:graph_execution_options",
        "//machina/core/tfrt/runtime",
        "@com_google_googletest//:gtest_main",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Parser",
        "@local_xla//xla/tsl/platform:statusor",
        "@tf_runtime//:bef",
        "@tf_runtime//:hostcontext",
    ],
)

filegroup(
    name = "testdata",
    srcs = glob(
        ["testdata/**"],
    ),
    visibility = ["//visibility:public"],
)

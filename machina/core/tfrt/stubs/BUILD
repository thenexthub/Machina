load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = [
        # copybara:uncomment "//learning/brain/experimental/tfrt:__subpackages__",
        # copybara:uncomment "//learning/brain/tfrt:__subpackages__",
        "//machina/compiler/mlir/tfrt:__subpackages__",
        "//machina/core/tfrt:__subpackages__",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "tfrt_native_lowering_stub",
    srcs = ["tfrt_native_lowering_stub.cc"],
    hdrs = ["tfrt_native_lowering_stub.h"],
    deps = [
        "//machina/core/tfrt/graph_executor:executable_context",
        "//machina/core/tfrt/graph_executor:sync_resource_state",
        "//machina/core/tfrt/mlrt/interpreter:context",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@toolchain-project//mlir:IR",
        "@tf_runtime//:hostcontext",
    ],
)

cc_library(
    name = "model_config_stub",
    srcs = ["model_config_stub.cc"],
    hdrs = ["model_config_stub.h"],
    deps = [
        "//machina/core/tfrt/runtime",
        "//machina/core/tfrt/saved_model:saved_model_util",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
    ],
)

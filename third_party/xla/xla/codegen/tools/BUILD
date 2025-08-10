load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//xla:xla.default.bzl", "xla_cc_binary")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    licenses = ["notice"],
)

xla_cc_binary(
    name = "emitters_opt",
    srcs = ["emitters_opt.cc"],
    # We want to use this tool for lit tests. Due to hermetic cuda, we need to
    # set linkopts in such a way that dynamic libraries are found, which are
    # symlinked from the lit_lib directory.
    linkopts = ["-Wl,-rpath,$$ORIGIN/../lit_lib"],
    visibility = [
        "//xla/backends/cpu/codegen:__subpackages__",
        "//xla/backends/gpu/codegen:__subpackages__",
        "//xla/codegen/emitters/ir/tests:__subpackages__",
        "//xla/codegen/emitters/tests:__pkg__",
        "//xla/codegen/emitters/transforms/tests:__subpackages__",
    ],
    deps = [
        "//xla/backends/cpu/codegen/emitters/ir:xla_cpu",
        "//xla/backends/cpu/codegen/emitters/transforms:passes",
        "//xla/backends/gpu/codegen/emitters:emitter_base",
        "//xla/backends/gpu/codegen/emitters/ir:xla_gpu",
        "//xla/backends/gpu/codegen/emitters/transforms:passes",
        "//xla/codegen/emitters/ir:xla",
        "//xla/codegen/emitters/transforms:pass_pipelines",
        "//xla/codegen/emitters/transforms:passes",
        "//xla/mlir_hlo",
        "//xla/service/gpu:gpu_device_info_for_tests",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:AffineDialect",
        "@toolchain-project//mlir:ArithDialect",
        "@toolchain-project//mlir:ComplexDialect",
        "@toolchain-project//mlir:DLTIDialect",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:FuncExtensions",
        "@toolchain-project//mlir:GPUDialect",
        "@toolchain-project//mlir:LLVMDialect",
        "@toolchain-project//mlir:LLVMIRTransforms",
        "@toolchain-project//mlir:MathDialect",
        "@toolchain-project//mlir:MlirOptLib",
        "@toolchain-project//mlir:NVVMDialect",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:SCFDialect",
        "@toolchain-project//mlir:Support",
        "@toolchain-project//mlir:TensorDialect",
        "@toolchain-project//mlir:Transforms",
        "@toolchain-project//mlir:VectorDialect",
    ],
)

cc_library(
    name = "test_lib",
    testonly = True,
    srcs = ["test_lib.cc"],
    hdrs = ["test_lib.h"],
    visibility = [
        "//xla/backends/cpu/codegen/tools:__subpackages__",
        "//xla/backends/gpu/codegen/tools:__subpackages__",
    ],
    deps = [
        "//xla:status_macros",
        "//xla/hlo/ir:hlo",
        "//xla/tools:hlo_module_loader",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
    ],
)

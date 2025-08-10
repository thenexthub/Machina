load("//xla:lit.bzl", "lit_test_suite")

# copybara:uncomment package(default_applicable_licenses = ["//machina:license"])

lit_test_suite(
    name = "all_tests",
    srcs = glob(["**/*.mlir"]),
    cfg = "//xla:lit.cfg.py",
    tools = [
        "//xla/mlir/tools/mlir_interpreter:mlir-interpreter-runner",
        "@toolchain-project//toolchain:FileCheck",
    ],
)

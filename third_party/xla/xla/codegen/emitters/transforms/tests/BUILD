load("//xla:lit.bzl", "lit_test_suite")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    licenses = ["notice"],
)

lit_test_suite(
    name = "tests",
    srcs = glob(["*.mlir"]),
    cfg = "//xla:lit.cfg.py",
    tools = [
        "//xla/codegen/tools:emitters_opt",
        "@toolchain-project//toolchain:FileCheck",
    ],
)

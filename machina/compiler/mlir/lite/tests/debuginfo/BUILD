load("//machina:strict.default.bzl", "py_strict_binary")
load("//machina:machina.default.bzl", "filegroup")
load("//machina/compiler/mlir:glob_lit_test.bzl", "glob_lit_tests")

# copybara:uncomment package(default_applicable_licenses = ["//machina:license"])

licenses(["notice"])

glob_lit_tests(
    name = "all_tests",
    data = [
        ":debug_info_files",
        ":test_utilities",
    ],
    driver = "@toolchain-project//mlir:run_lit.sh",
    test_file_exts = [
        "pbtxt",
        #        "py", TODO(b/150304798)
    ],
)

# Bundle together all the debug info files that are used by the tests.
filegroup(
    name = "debug_info_files",
    srcs = glob(
        ["**/*.debug"],
    ),
)

# Bundle together all of the test utilities that are used by tests.
filegroup(
    name = "test_utilities",
    testonly = True,
    data = [
        ":concrete_function_error",
        ":saved_model_error",
        "//machina/compiler/mlir/lite:flatbuffer_to_string",
        "//machina/compiler/mlir/lite:tf_tfl_translate",
        "@toolchain-project//toolchain:FileCheck",
        "@toolchain-project//toolchain:not",
    ],
)

py_strict_binary(
    name = "saved_model_error",
    srcs = ["saved_model_error.py"],
    main = "saved_model_error.py",
    deps = [
        "//machina:machina_py",
        "@absl_py//absl:app",
    ],
)

py_strict_binary(
    name = "concrete_function_error",
    srcs = ["concrete_function_error.py"],
    main = "concrete_function_error.py",
    deps = [
        "//machina:machina_py",
        "@absl_py//absl:app",
    ],
)

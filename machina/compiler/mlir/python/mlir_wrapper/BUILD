load("//machina:machina.default.bzl", "tf_python_pybind_extension")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    licenses = ["notice"],
)

tf_python_pybind_extension(
    name = "mlir_wrapper",
    srcs = [
        "attrs.cc",
        "basic_classes.cc",
        "builders.cc",
        "mlir_wrapper.cc",
        "mlir_wrapper.h",
        "ops.cc",
        "types.cc",
    ],
    enable_stub_generation = True,
    pytype_srcs = [
        "mlir_wrapper.pyi",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//machina/compiler/mlir/machina",
        "//machina/compiler/mlir/machina:machina_types",
        "//machina/python/lib/core:pybind11_lib",
        "//machina/python/lib/core:pybind11_status",
        "@toolchain-project//toolchain:FileCheckLib",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Parser",
        "@pybind11",
    ],
)

tf_python_pybind_extension(
    name = "filecheck_wrapper",
    srcs = ["filecheck_wrapper.cc"],
    enable_stub_generation = True,
    pytype_srcs = [
        "filecheck_wrapper.pyi",
    ],
    starlark_only = True,
    visibility = ["//visibility:public"],
    deps = [
        "//machina/python/lib/core:pybind11_lib",
        "//machina/python/lib/core:pybind11_status",
        "@toolchain-project//toolchain:FileCheckLib",
        "@toolchain-project//toolchain:Support",
        "@pybind11",
    ],
)

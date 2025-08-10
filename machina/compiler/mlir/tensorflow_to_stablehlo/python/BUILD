load("@local_xla//xla/tsl/platform:build_config_root.bzl", "if_pywrap")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load(
    "//machina:machina.default.bzl",
    "get_compatible_with_portable",
    "tf_py_strict_test",
    "tf_python_pybind_extension",
)
load("//machina/compiler/mlir/quantization/stablehlo:internal_visibility_allowlist.bzl", "internal_visibility_allowlist")

package_group(
    name = "internal_visibility_allowlist_package",
    packages = [
        "//machina/compiler/mlir/lite/...",
        "//machina/compiler/mlir/quantization/...",
        "//machina/compiler/mlir/tf2xla/transforms/...",
        "//machina/lite/...",
    ] + internal_visibility_allowlist(),
)

package(
    # copybara:uncomment default_applicable_licenses = ["@stablehlo//:license"],
    default_visibility = [
        ":internal_visibility_allowlist_package",
        "//machina:__pkg__",
        "//machina/python:__pkg__",
    ],
    licenses = ["notice"],
)

# copybara:uncomment_begin(google-only)
# tf_py_strict_test(
#     name = "machina_to_stablehlo_test",
#     testonly = 1,
#     srcs = ["integration_test/machina_to_stablehlo_test.py"],
#     deps = [
#         ":pywrap_machina_to_stablehlo",
#         "//testing/pymocks:matchers",
#         "//third_party/py/mlir",
#         "//third_party/py/mlir:ir",
#         "//third_party/py/mlir:stablehlo_dialect",
#         "//third_party/py/mlir/_mlir_libs:_mlirRegisterEverything",
#         "//machina:machina_py",
#         "//machina/compiler/mlir/stablehlo",
#         "//machina/python/framework:test_lib",
#         "//machina/python/platform:client_testlib",
#         "//machina/python/types:core",
#     ],
# )
# copybara:uncomment_end

# This is a header-only target. The purpose of `pywrap_machina_to_stablehlo_lib_*` targets is to expose only
# the symbols that are required by `pywrap_machina_to_stablehlo` that translates them to python functions.
# The only intended use case of this library is by `pywrap_machina_to_stablehlo`. Not letting
# `pywrap_machina_to_stablehlo` directly depend on sub-libraries like `static_range_srq` and instead haiving
# a consolidated impl library `pywrap_machina_to_stablehlo_lib_impl` allows the maintainers to avoid
# declaring multiple impl libraries to `libmachina_cc` and `lib_pywrap_machina_internal`,
# which is required to avoid ODR violations.
cc_library(
    name = "pywrap_machina_to_stablehlo_lib_header_only",
    srcs = [],
    hdrs = ["pywrap_machina_to_stablehlo_lib.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = ["//visibility:private"],  # ONLY for `pywrap_machina_to_stablehlo`.
    deps = [
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
    ],
)

# See the comments for `pywrap_machina_to_stablehlo_lib_header_only`.
cc_library(
    name = "pywrap_machina_to_stablehlo_lib_impl",
    srcs = ["pywrap_machina_to_stablehlo_lib.cc"],
    hdrs = ["pywrap_machina_to_stablehlo_lib.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = [
        "//machina:__pkg__",  # For libmachina_cc.so.
        "//machina/python:__pkg__",  # For lib_pywrap_machina_internal.so.
    ],
    deps = [
        "//machina/compiler/mlir/machina",
        "//machina/compiler/mlir/machina_to_stablehlo:tf_to_stablehlo",
        "//machina/core:lib",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:BytecodeWriter",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Parser",
        "@toolchain-project//mlir:Support",
        "@local_xla//third_party/python_runtime:headers",
    ],
)

tf_python_pybind_extension(
    name = "pywrap_machina_to_stablehlo",
    srcs = ["pywrap_machina_to_stablehlo.cc"],
    pytype_srcs = ["pywrap_machina_to_stablehlo.pyi"],
    # Each dependency MUST be either header-only or exclusive.
    deps = [
        "@com_google_absl//absl/strings:string_view",
        "@local_xla//third_party/python_runtime:headers",
        "@pybind11",
        "@pybind11_abseil//pybind11_abseil:absl_casters",
        "@pybind11_abseil//pybind11_abseil:status_casters",
    ] + if_pywrap(
        if_false = [":pywrap_machina_to_stablehlo_lib_header_only"],
        if_true = [":pywrap_machina_to_stablehlo_lib_impl"],
    ),
)

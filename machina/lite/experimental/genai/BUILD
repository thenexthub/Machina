load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

# Custom Ops useful for GenAI models.
load("//machina:machina.default.bzl", "pybind_extension")
load("//machina/lite:build_def.bzl", "tflite_copts")
load("//machina/lite:special_rules.bzl", "tflite_portable_test_suite")

# copybara:uncomment package(default_applicable_licenses = ["//machina:license"])

cc_library(
    name = "genai_ops",
    srcs = [
        "external_kvcache.cc",
        "genai_ops.cc",
        "kvcache.cc",
        "sdpa.cc",
    ],
    hdrs = [
        "genai_ops.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//machina/lite:framework",
        "//machina/lite/c:c_api_types",
        "//machina/lite/c:common",
        "//machina/lite/core:subgraph",
        "//machina/lite/core/c:common",
        "//machina/lite/experimental/resource",
        "//machina/lite/experimental/resource:cache_buffer",
        "//machina/lite/kernels:kernel_util",
        "//machina/lite/kernels/internal:common",
        "//machina/lite/kernels/internal:reference_base",
        "//machina/lite/kernels/internal:tensor",
        "//machina/lite/kernels/internal:types",
        "@flatbuffers",
    ],
)

cc_test(
    name = "kvcache_test",
    srcs = ["kvcache_test.cc"],
    copts = tflite_copts(),
    deps = [
        ":genai_ops",
        "//machina/lite/c:c_api_types",
        "//machina/lite/kernels:test_util",
        "//machina/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "external_kvcache_test",
    srcs = ["external_kvcache_test.cc"],
    copts = tflite_copts(),
    deps = [
        ":genai_ops",
        "//machina/lite:framework",
        "//machina/lite:util",
        "//machina/lite/c:c_api_types",
        "//machina/lite/c:common",
        "//machina/lite/kernels:test_util",
        "//machina/lite/kernels/internal:tensor",
        "//machina/lite/kernels/internal:types",
        "//machina/lite/schema:schema_fbs",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

pybind_extension(
    name = "pywrap_genai_ops",
    srcs = [
        "genai_ops_wrapper.cc",
    ],
    hdrs = ["genai_ops.h"],
    additional_exported_symbols = ["GenAIOpsRegisterer"],
    common_lib_packages = [
        "litert/python",
        "machina/lite/python",
    ],
    enable_stub_generation = True,
    link_in_framework = True,
    module_name = "pywrap_genai_ops",
    pytype_srcs = [
        "pywrap_genai_ops.pyi",
    ],
    visibility = ["//visibility:public"],
    wrap_py_init = True,
    deps = [
        ":genai_ops",
        "//machina/lite:framework_stable",
        "//machina/lite:mutable_op_resolver",
        "//machina/lite/c:common",
        "@local_xla//third_party/python_runtime:headers",
        "@pybind11",
    ],
)

tflite_portable_test_suite()

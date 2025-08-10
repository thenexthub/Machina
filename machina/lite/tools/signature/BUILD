load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("//machina:strict.default.bzl", "py_strict_library", "py_strict_test")
load("//machina:machina.default.bzl", "get_compatible_with_portable", "pybind_extension", "replace_with_portable_tf_lib_when_required")

# Utilities for signature_defs in TFLite
load("//machina/lite:build_def.bzl", "tflite_copts")
load("//machina/lite:special_rules.bzl", "tflite_portable_test_suite")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "signature_def_util",
    srcs = ["signature_def_util.cc"],
    hdrs = ["signature_def_util.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts(),
    features = select({
        "//machina:android": ["-layering_check"],
        "//conditions:default": [],
    }),
    deps = replace_with_portable_tf_lib_when_required([
        "//machina/core:lib_proto_parsing",
        "//machina/core:protos_all_cc",
        "//machina/core/platform:errors",
        "//machina/core/platform:status",
    ]) + [
        "//machina/core:protos_all_cc_impl",
        "//machina/lite:framework",
        "//machina/lite/core:framework",
        "//machina/lite/core/c:common",
        "//machina/lite/schema:schema_fbs",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_protobuf//:protobuf",
        "@flatbuffers",
        "@local_tsl//tsl/platform:status",
    ],
)

cc_test(
    name = "signature_def_util_test",
    size = "small",
    srcs = ["signature_def_util_test.cc"],
    data = [
        "//machina/lite:testdata/add.bin",
    ],
    tags = [
        "no_oss",
        "tflite_not_portable",
    ],
    deps = [
        ":signature_def_util",
        "//machina/cc/saved_model:signature_constants",
        "//machina/core/platform:errors",
        "//machina/core/protobuf:for_core_protos_cc",
        "//machina/lite:framework",
        "//machina/lite:string",
        "//machina/lite/c:c_api",
        "//machina/lite/core:framework",
        "//machina/lite/core/c:common",
        "//machina/lite/schema:schema_fbs",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
        "@flatbuffers//:runtime_cc",
        "@local_tsl//tsl/platform:status",
    ],
)

pybind_extension(
    name = "_pywrap_signature_def_util_wrapper",
    srcs = [
        "signature_def_util_wrapper_pybind11.cc",
    ],
    data = [
        "_pywrap_signature_def_util_wrapper.pyi",
    ],
    enable_stub_generation = True,
    deps = [
        ":signature_def_util",
        "//machina/core/protobuf:for_core_protos_cc",
        "//machina/lite:framework",
        "//machina/lite/core:framework",
        "//machina/python/lib/core:pybind11_lib",
        "@com_google_absl//absl/status",
        "@pybind11",
    ],
)

py_strict_library(
    name = "signature_def_utils",
    srcs = ["signature_def_utils.py"],
    deps = [
        ":_pywrap_signature_def_util_wrapper",
        "//machina/core:protos_all_py",
    ],
)

py_strict_test(
    name = "signature_def_utils_test",
    srcs = ["signature_def_utils_test.py"],
    data = ["//machina/lite:testdata/add.bin"],
    tags = [
        "no_mac",
        "no_oss",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":signature_def_utils",
        #internal proto upb dep
        "//machina:machina_py",
        "//machina/core:protos_all_py",
    ],
)

tflite_portable_test_suite()

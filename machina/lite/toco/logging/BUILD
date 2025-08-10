load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//machina:strict.default.bzl", "py_strict_library", "py_strict_test")

# Placeholder: load py_proto_library
load(
    "//machina:machina.bzl",
    "tf_cc_test",
)
load("//machina:machina.default.bzl", "get_compatible_with_portable")
load(
    "//machina/core/platform:build_config.bzl",
    "tf_proto_library",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

tf_proto_library(
    name = "toco_conversion_log_proto",
    srcs = ["toco_conversion_log.proto"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "conversion_log_util",
    srcs = ["conversion_log_util.cc"],
    hdrs = ["conversion_log_util.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = ["//visibility:public"],
    deps = [
        ":toco_conversion_log_proto_cc",
        "//machina/core:protos_all_cc",
        "//machina/lite:version",
        "//machina/lite/toco:model",
        "//machina/lite/toco:tooling_util",
        "//machina/lite/toco/tflite:export",
        "//machina/lite/toco/tflite:operator",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
    ],
)

tf_cc_test(
    name = "conversion_log_util_test",
    srcs = ["conversion_log_util_test.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":conversion_log_util",
        "//machina/core:ops",
        "//machina/core:protos_all_cc",
        "//machina/lite/toco:model",
        "//machina/lite/toco:model_flags_proto_cc",
        "@com_google_absl//absl/memory",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

filegroup(
    name = "html_template",
    srcs = [
        "template.html",
    ],
)

py_strict_library(
    name = "gen_html",
    srcs = ["gen_html.py"],
    data = [
        "html_template",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":toco_conversion_log_proto_py",
        "//machina/python/lib/io:file_io",
        "//machina/python/platform:resource_loader",
    ],
)

py_strict_test(
    name = "gen_html_test",
    srcs = ["gen_html_test.py"],
    data = [
        ":template.html",
        "//machina/lite/toco/logging/testdata:generated.html",
        "//machina/lite/toco/logging/testdata:toco_log_after.pb",
        "//machina/lite/toco/logging/testdata:toco_log_before.pb",
        "//machina/lite/toco/logging/testdata:toco_tf_graph.dot",
        "//machina/lite/toco/logging/testdata:toco_tflite_graph.dot",
    ],
    deps = [
        ":gen_html",
        ":toco_conversion_log_proto_py",
        #internal proto upb dep
        "//machina/python/framework:test_lib",
        "//machina/python/lib/io:file_io",
        "//machina/python/platform:client_testlib",
        "//machina/python/platform:resource_loader",
    ],
)

# copybara:uncomment_begin(google-only)
# py_proto_library(
#     name = "toco_conversion_log_proto_py",
#     visibility = ["//visibility:public"],
#     deps = [":toco_conversion_log_proto"],
# )
# copybara:uncomment_end

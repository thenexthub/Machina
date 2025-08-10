load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("//machina/lite:special_rules.bzl", "tflite_portable_test_suite")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "preprocessor",
    srcs = ["preprocessor.cc"],
    hdrs = ["preprocessor.h"],
    deps = [
        "//machina/lite/delegates/gpu/common:status",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "preprocessor_test",
    srcs = ["preprocessor_test.cc"],
    tags = [
        "local",
        "tflite_not_portable_ios",
    ],
    deps = [
        ":preprocessor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "object_accessor",
    srcs = ["object_accessor.cc"],
    hdrs = ["object_accessor.h"],
    deps = [
        ":preprocessor",
        ":variable_accessor",
        "//machina/lite/delegates/gpu/common:access_type",
        "//machina/lite/delegates/gpu/common:data_type",
        "//machina/lite/delegates/gpu/common:types",
        "//machina/lite/delegates/gpu/gl:object",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:variant",
    ],
)

cc_test(
    name = "object_accessor_test",
    srcs = ["object_accessor_test.cc"],
    tags = [
        "local",
    ],
    deps = [
        ":object_accessor",
        ":preprocessor",
        ":variable_accessor",
        "//machina/lite/delegates/gpu/common:types",
        "//machina/lite/delegates/gpu/gl:object",
        "//machina/lite/delegates/gpu/gl:variable",
        "@com_google_absl//absl/types:variant",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "shader_code",
    hdrs = ["shader_code.h"],
    deps = [
        "//machina/lite/delegates/gpu/common:model",
        "//machina/lite/delegates/gpu/common:types",
        "//machina/lite/delegates/gpu/gl:object",
        "//machina/lite/delegates/gpu/gl:variable",
    ],
)

cc_library(
    name = "shader_codegen",
    srcs = ["shader_codegen.cc"],
    hdrs = ["shader_codegen.h"],
    deps = [
        ":compiled_node",
        ":object_accessor",
        ":preprocessor",
        ":shader_code",
        ":variable_accessor",
        "//machina/lite/delegates/gpu/common:gpu_info",
        "//machina/lite/delegates/gpu/common:model",
        "//machina/lite/delegates/gpu/common:status",
        "//machina/lite/delegates/gpu/gl:compiler_options",
        "//machina/lite/delegates/gpu/gl:object",
        "//machina/lite/delegates/gpu/gl:variable",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "compiled_node",
    srcs = ["compiled_node.cc"],
    hdrs = ["compiled_node.h"],
    deps = [
        ":rename",
        "//machina/lite/delegates/gpu/common:model",
        "//machina/lite/delegates/gpu/common:status",
        "//machina/lite/delegates/gpu/gl:node_shader",
        "//machina/lite/delegates/gpu/gl:object",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "compiled_node_test",
    srcs = ["compiled_node_test.cc"],
    deps = [
        ":compiled_node",
        "//machina/lite/delegates/gpu/gl:variable",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "fuse_inplace",
    srcs = ["fuse_inplace.cc"],
    hdrs = ["fuse_inplace.h"],
    deps = [
        ":compiled_node",
        ":preprocessor",
        "//machina/lite/delegates/gpu/common:model",
        "//machina/lite/delegates/gpu/common:model_transformer",
        "//machina/lite/delegates/gpu/common:types",
        "//machina/lite/delegates/gpu/gl:node_shader",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:any",
    ],
)

cc_library(
    name = "fuse_inline",
    srcs = ["fuse_inline.cc"],
    hdrs = ["fuse_inline.h"],
    deps = [
        ":compiled_node",
        ":shader_code",
        "//machina/lite/delegates/gpu/common:model",
        "//machina/lite/delegates/gpu/common:model_transformer",
        "//machina/lite/delegates/gpu/common:status",
        "//machina/lite/delegates/gpu/common:types",
        "//machina/lite/delegates/gpu/gl:node_shader",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:any",
    ],
)

cc_library(
    name = "rename",
    srcs = ["rename.cc"],
    hdrs = ["rename.h"],
    deps = [
        ":object_accessor",
        ":preprocessor",
        ":variable_accessor",
        "//machina/lite/delegates/gpu/common:status",
        "//machina/lite/delegates/gpu/gl:node_shader",
        "//machina/lite/delegates/gpu/gl:object",
        "//machina/lite/delegates/gpu/gl:variable",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "fuse_auto_input",
    srcs = ["fuse_auto_input.cc"],
    hdrs = ["fuse_auto_input.h"],
    deps = [
        ":compiled_node",
        "//machina/lite/delegates/gpu/common:model",
        "//machina/lite/delegates/gpu/common:model_transformer",
        "//machina/lite/delegates/gpu/common:operations",
        "//machina/lite/delegates/gpu/common:types",
        "//machina/lite/delegates/gpu/gl:node_shader",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:any",
        "@com_google_absl//absl/types:variant",
    ],
)

cc_test(
    name = "fuse_auto_input_test",
    srcs = ["fuse_auto_input_test.cc"],
    tags = [
        "local",
        "no_mac",  # TODO(b/171881489)
        "no_oss",  # TODO(b/171881489)
    ],
    deps = [
        ":compiled_node",
        ":fuse_auto_input",
        "//machina/lite/delegates/gpu/common:model",
        "//machina/lite/delegates/gpu/common:model_transformer",
        "//machina/lite/delegates/gpu/gl:node_shader",
        "@com_google_absl//absl/types:any",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "variable_accessor",
    srcs = ["variable_accessor.cc"],
    hdrs = ["variable_accessor.h"],
    deps = [
        ":preprocessor",
        "//machina/lite/delegates/gpu/common:types",
        "//machina/lite/delegates/gpu/gl:variable",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:variant",
    ],
)

cc_test(
    name = "variable_accessor_test",
    srcs = ["variable_accessor_test.cc"],
    tags = [
        "local",
        "tflite_not_portable_ios",
    ],
    deps = [
        ":preprocessor",
        ":variable_accessor",
        "//machina/lite/delegates/gpu/common:types",
        "@com_google_googletest//:gtest_main",
    ],
)

tflite_portable_test_suite()

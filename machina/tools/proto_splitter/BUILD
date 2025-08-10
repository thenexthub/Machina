# Description:
#   Utilities for splitting and joining large protos > 2GB.

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//machina:pytype.default.bzl", "pytype_strict_library")
load("//machina:strict.default.bzl", "py_strict_test")

# Placeholder: load py_proto_library
load(
    "//machina:machina.bzl",
    "if_oss",
    "tf_cc_test",
)
load(
    "//machina/core/platform:build_config.bzl",
    "tf_proto_library",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = [
        "__subpackages__",
        "//machina:internal",
    ],
    licenses = ["notice"],
)

tf_proto_library(
    name = "versions_proto",
    srcs = ["versions.proto"],
    make_default_target_header_only = True,
)

tf_proto_library(
    name = "chunk_proto",
    srcs = ["chunk.proto"],
    make_default_target_header_only = True,
    protodeps = [":versions_proto"],
)

# OSS-only: Required to link the impl files.
cc_library(
    name = "protos_impl",
    deps = if_oss([
        "//machina/tools/proto_splitter:chunk_proto_cc_impl",
        "//machina/tools/proto_splitter:versions_proto_cc_impl",
        "//machina/tools/proto_splitter/testdata:test_message_proto_cc_impl",
    ]),
)

# copybara:uncomment_begin(google-only)
#
# py_proto_library(
#     name = "versions_proto_py_pb2",
#     deps = [
#         ":versions_proto",
#     ],
# )
#
# py_proto_library(
#     name = "chunk_proto_py_pb2",
#     deps = [
#         ":chunk_proto",
#     ],
# )
#
# copybara:uncomment_end

pytype_strict_library(
    name = "split",
    srcs = ["split.py"],
    deps = [
        ":chunk_proto_py",
        ":util",
        ":version",
        ":versions_proto_py",
        "//machina/python/lib/io:file_io",
        "@absl_py//absl/logging",
        "@riegeli_py//python/riegeli",
    ],
)

pytype_strict_library(
    name = "version",
    srcs = ["version.py"],
)

py_strict_test(
    name = "split_test",
    srcs = ["split_test.py"],
    tags = [
        "no_mac",  # b/291933687
        "no_windows",  # b/291001524
    ],
    deps = [
        ":chunk_proto_py",
        ":split",
        ":versions_proto_py",
        #internal proto upb dep
        "//machina/python/platform:client_testlib",
        "//machina/tools/proto_splitter/testdata:test_message_proto_py",
        "@absl_py//absl/testing:parameterized",
        "@riegeli_py//python/riegeli",
    ],
)

pytype_strict_library(
    name = "constants",
    srcs = ["constants.py"],
    visibility = ["//visibility:public"],
)

pytype_strict_library(
    name = "split_graph_def",
    srcs = ["split_graph_def.py"],
    deps = [
        ":constants",
        ":split",
        ":util",
        "//machina/core:protos_all_py",
        "//machina/python/framework:tensor_util",
    ],
)

py_strict_test(
    name = "split_graph_def_test",
    srcs = ["split_graph_def_test.py"],
    tags = [
        "no_mac",  # b/291933687
        "no_windows",  # b/291001524
    ],
    deps = [
        ":chunk_proto_py",
        ":constants",
        ":split_graph_def",
        ":util",
        #internal proto upb dep
        "//machina/core:protos_all_py",
        "//machina/python/platform:client_testlib",
        "//machina/tools/proto_splitter/python:test_util",
    ],
)

pytype_strict_library(
    name = "util",
    srcs = ["util.py"],
    deps = [":chunk_proto_py"],
)

py_strict_test(
    name = "util_test",
    srcs = ["util_test.py"],
    deps = [
        ":util",
        #internal proto upb dep
        "//machina/python/platform:client_testlib",
        "//machina/tools/proto_splitter/testdata:test_message_proto_py",
    ],
)

cc_library(
    name = "merge_impl",
    srcs = [
        "merge.cc",
        "merge.h",
    ],
    deps = [
        ":chunk_proto_cc",
        "//machina/core:lib",
        "//machina/tools/proto_splitter/cc:util",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:protobuf",
        "@riegeli//riegeli/base:object",
        "@riegeli//riegeli/bytes:fd_reader",
        "@riegeli//riegeli/bytes:string_reader",
        "@riegeli//riegeli/records:record_reader",
    ] + if_oss([
        "//machina/tools/proto_splitter:protos_impl",
    ]),
)

cc_library(
    name = "merge",
    hdrs = ["merge.h"],
    visibility = [
        "//platforms/darwinn/contrib/waymo:__subpackages__",
        "//machina:internal",
        "//machina/cc/experimental/tf2:__subpackages__",
        "//machina/cc/saved_model/image_format:__subpackages__",
        "//waymo/onboard/ml/chauffeur_net/utils:__subpackages__",
    ],
    deps = [
        ":chunk_proto_cc",
        ":merge_impl",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:string_view",
        "@local_tsl//tsl/platform:protobuf",
        "@riegeli//riegeli/bytes:fd_reader",
        "@riegeli//riegeli/records:record_reader",
    ] + if_oss([
        "//machina/tools/proto_splitter:protos_impl",
    ]),
)

tf_cc_test(
    name = "merge_test",
    srcs = ["merge_test.cc"],
    data = [
        "//machina/cc/saved_model:testdata/chunked_saved_model/chunked_model/saved_model.cpb",
        "//machina/cc/saved_model:testdata/chunked_saved_model/chunked_model/saved_model.pbtxt",
        "//machina/tools/proto_splitter/testdata:bf-split-tree.cpb",
        "//machina/tools/proto_splitter/testdata:bf-split-tree.pbtxt",
        "//machina/tools/proto_splitter/testdata:df-split-tree.cpb",
        "//machina/tools/proto_splitter/testdata:df-split-tree.pbtxt",
        "//machina/tools/proto_splitter/testdata:many-field.cpb",
        "//machina/tools/proto_splitter/testdata:many-field.pbtxt",
        "//machina/tools/proto_splitter/testdata:split-large-constant.cpb",
        "//machina/tools/proto_splitter/testdata:split-large-constant.pbtxt",
        "//machina/tools/proto_splitter/testdata:split-large-nodes.cpb",
        "//machina/tools/proto_splitter/testdata:split-large-nodes.pbtxt",
        "//machina/tools/proto_splitter/testdata:split-lots-nodes.cpb",
        "//machina/tools/proto_splitter/testdata:split-lots-nodes.pbtxt",
        "//machina/tools/proto_splitter/testdata:split-standard.cpb",
        "//machina/tools/proto_splitter/testdata:split-standard.pbtxt",
        "//machina/tools/proto_splitter/testdata:split-tree.pbtxt",
    ],
    tags = [
        "no_mac",  # b/291933687
        "no_windows",  # b/291001524
    ],
    deps = [
        ":chunk_proto_cc",
        ":merge",
        "//machina/core:protos_all_cc",
        "//machina/core:test",
        "//machina/core/platform:path",
        "//machina/tools/proto_splitter/cc:test_util",
        "//machina/tools/proto_splitter/cc:util",
        "//machina/tools/proto_splitter/testdata:test_message_proto_cc",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:protobuf",
        "@local_tsl//tsl/platform:statusor",
        "@local_xla//xla/tsl/platform:env",
    ] + if_oss([
        "//machina/tools/proto_splitter:protos_impl",
    ]),
)

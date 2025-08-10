load("@rules_cc//cc:cc_library.bzl", "cc_library")

# Placeholder: load py_proto_library
load("//machina:machina.bzl", "tf_cc_test")
load("//machina/core/platform:build_config.bzl", "tf_proto_library")

DEFAULT_VISIBILITY = [
    "//machina/core/grappler/optimizers/inference:__subpackages__",
    "//learning/serving:__subpackages__",
    "//machina_serving:__subpackages__",
    "//machina:__subpackages__",
]

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = DEFAULT_VISIBILITY,
    licenses = ["notice"],
)

# Expand the DEFAULT_VISIBILITY so that we can replace with public visibility with copybara.
tf_proto_library(
    name = "batch_op_rewriter_proto",
    srcs = ["batch_op_rewriter.proto"],
    visibility = ["//visibility:public"],
)

# copybara:uncomment_begin(google-only)
# py_proto_library(
#     name = "batch_op_rewriter_proto_py_pb2",
#     visibility = ["//visibility:public"],
#     deps = [":batch_op_rewriter_proto"],
# )
# copybara:uncomment_end

cc_library(
    name = "batch_op_rewriter",
    srcs = ["batch_op_rewriter.cc"],
    hdrs = ["batch_op_rewriter.h"],
    deps = [
        ":batch_op_rewriter_proto_cc",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core/grappler:grappler_item",
        "//machina/core/grappler/optimizers:custom_graph_optimizer",
        "//machina/core/grappler/optimizers:custom_graph_optimizer_registry",
        "//machina/tools/graph_transforms:transform_utils",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_protobuf//:protobuf",
        "@com_google_protobuf//:protobuf_headers",
    ],
    alwayslink = 1,
)

tf_cc_test(
    name = "batch_op_rewriter_test",
    srcs = ["batch_op_rewriter_test.cc"],
    deps = [
        ":batch_op_rewriter",
        ":batch_op_rewriter_proto_cc",
        "//machina/core:protos_all_cc",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core/grappler:grappler_item",
        "//machina/core/grappler/optimizers:graph_optimizer",
        "//machina/core/platform:errors",
        "//machina/tools/graph_transforms:transform_utils",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_protobuf//:protobuf",
    ],
)

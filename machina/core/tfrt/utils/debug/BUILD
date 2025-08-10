load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//machina:machina.bzl", "tf_cc_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    licenses = ["notice"],
)

cc_library(
    name = "node_io_dump_rewriter",
    srcs = ["node_io_dump_rewriter.cc"],
    hdrs = ["node_io_dump_rewriter.h"],
    deps = [
        "//machina/core:core_cpu_base",
        "//machina/core:framework",
        "//machina/core/lib/core:errors",
        "//machina/core/platform:status",
        "//machina/core/protobuf:for_core_protos_cc",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@local_xla//xla/tsl/platform:errors",
        "@local_xla//xla/tsl/platform:statusor",
    ],
)

# copybara:uncomment_begin(Google Internal Test)
# tf_cc_test(
#     name = "node_io_dump_rewriter_test",
#     srcs = ["node_io_dump_rewriter_test.cc"],
#     data = [
#         "//machina/core/tfrt/saved_model/tests:toy_v1/1/saved_model.pb",
#         "//machina/core/tfrt/saved_model/tests:toy_v1/1/variables/variables.data-00000-of-00001",
#         "//machina/core/tfrt/saved_model/tests:toy_v1/1/variables/variables.index",
#         "//machina/core/tfrt/saved_model/tests:toy_v2/saved_model.pb",
#         "//machina/core/tfrt/saved_model/tests:toy_v2/variables/variables.data-00000-of-00001",
#         "//machina/core/tfrt/saved_model/tests:toy_v2/variables/variables.index",
#     ],
#     deps = [
#         ":node_io_dump_rewriter",
#         "@com_google_googletest//:gtest_main",
#         "@com_google_absl//absl/log:check",
#         "@com_google_absl//absl/status",
#         "@com_google_absl//absl/strings",
#         "//machina/cc:array_ops",
#         "//machina/cc:cc_ops",
#         "//machina/cc:scope",
#         "//machina/cc:sendrecv_ops",
#         "//machina/cc/saved_model:reader",
#         "@local_xla//xla/tsl/platform:statusor",
#         "@local_xla//xla/tsl/platform:test",
#         "//machina/core:core_cpu_base",
#         "//machina/core:portable_gif_internal",
#         "//machina/core:test",
#         "//machina/core/framework:op",
#         "//machina/core/framework:tensor",
#         "//machina/core/framework:types_proto_cc",
#         "//machina/core/platform:env",
#         "//machina/core/platform:resource_loader",
#         "//machina/core/protobuf:for_core_protos_cc",
#         "//machina/core/tfrt/saved_model:saved_model_cpu",
#         "//machina/core/tfrt/saved_model:saved_model_testutil",
#         "//machina/python/framework:test_ops_kernels",
#         "@local_tsl//tsl/platform:path",
#     ],
# )
# copybara:uncomment_end

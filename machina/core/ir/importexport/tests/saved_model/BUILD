# Description: Tests for SavedModel model roundtrip via MLIR TFG module

load("//machina:machina.bzl", "tf_cc_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    licenses = ["notice"],
)

filegroup(
    name = "saved_model_test_files",
    srcs = [
        "savedmodel_v1/saved_model.pb",
        "savedmodel_v2/saved_model.pb",
    ],
)

tf_cc_test(
    name = "saved_model_roundtrip_test",
    srcs = ["saved_model_roundtrip_test.cc"],
    data = [
        ":saved_model_test_files",
    ],
    deps = [
        "//machina/core:ops",
        "//machina/core:test",
        "//machina/core/framework:graph_debug_info_proto_cc",
        "//machina/core/ir/importexport:savedmodel_export",
        "//machina/core/ir/importexport:savedmodel_import",
        "//machina/core/ir/importexport/tests/roundtrip",
        "//machina/core/platform:env",
        "//machina/core/platform:path",
        "//machina/core/protobuf:for_core_protos_cc",
        "@com_google_googletest//:gtest_main",
    ],
)

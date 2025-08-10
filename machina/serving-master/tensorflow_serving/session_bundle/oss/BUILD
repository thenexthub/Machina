load("@org_machina//machina/core/platform:build_config.bzl", "tf_proto_library")
load("//machina_serving:serving.bzl", "oss_only_cc_test")

licenses(["notice"])

package(
    default_visibility = [
        "//machina_serving:internal",
    ],
    features = ["-layering_check"],
)

tf_proto_library(
    name = "manifest_proto",
    srcs = ["manifest.proto"],
)

cc_library(
    name = "session_bundle",
    hdrs = ["session_bundle.h"],
    deps = [
        "@org_machina//machina/core:core_cpu",
    ],
)

cc_library(
    name = "session_bundle_util",
    srcs = ["session_bundle_util.cc"],
    deps = [
        ":manifest_proto_cc",
        ":session_bundle",
        "//machina_serving/session_bundle:session_bundle_util_header",
        "@org_machina//machina/cc/saved_model:loader",
        "@org_machina//machina/core:lib",
    ],
)

oss_only_cc_test(
    name = "session_bundle_util_test",
    srcs = ["session_bundle_util_test.cc"],
    data = [
        "//machina_serving/session_bundle:session_bundle_half_plus_two",
        "@org_machina//machina/cc/saved_model:saved_model_half_plus_two",
    ],
    deps = [
        ":session_bundle_util",
        "//machina_serving/core/test_util:test_main",
        "//machina_serving/test_util",
    ],
)

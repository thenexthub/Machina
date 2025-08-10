load("//machina_serving:serving.bzl", "if_google", "if_oss")

licenses(["notice"])

package(default_visibility = [
    "//machina_serving:internal",
])

filegroup(
    name = "session_bundle_half_plus_two",
    srcs = glob([
        "testdata/half_plus_two/**",
    ]),
)

cc_library(
    name = "manifest_proto_header",
    hdrs = ["manifest_proto.h"],
    deps = ["//machina_serving/util:oss_or_google"] + if_oss([
        "//machina_serving/session_bundle/oss:manifest_proto_cc",
    ]) + if_google(["//machina_serving/session_bundle/google:manifest_proto_cc"]),
)

cc_library(
    name = "session_bundle",
    hdrs = ["session_bundle.h"],
    deps = ["//machina_serving/util:oss_or_google"] + if_oss([
        "//machina_serving/session_bundle/oss:session_bundle",
    ]) + if_google(["@org_machina//machina/contrib/session_bundle"]),
)

cc_library(
    name = "session_bundle_util_header",
    hdrs = ["session_bundle_util.h"],
    deps = [
        ":manifest_proto_header",
        ":session_bundle",
        "@org_machina//machina/cc/saved_model:loader",
    ],
)

cc_library(
    name = "session_bundle_util",
    deps = if_oss(["//machina_serving/session_bundle/oss:session_bundle_util"]) + if_google(["//machina_serving/session_bundle/google:session_bundle_util"]),
)

cc_library(
    name = "graph_rewriter",
    hdrs = ["graph_rewriter.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/synchronization",
        "@org_machina//machina/core:protos_all_cc",
        "@org_machina//machina/core/platform:errors",
        "@org_machina//machina/core/platform:status",
    ],
)

cc_library(
    name = "saved_model_config",
    srcs = ["saved_model_config.cc"],
    hdrs = ["saved_model_config.h"],
    deps = [
        "//machina_serving/servables/machina:saved_model_config_cc_proto",
        "//machina_serving/servables/machina:saved_model_config_util",
        "@com_google_absl//absl/status:statusor",
        "@org_machina//machina/core:core_cpu",
        "@org_machina//machina/core:lib",
    ],
)

cc_test(
    name = "saved_model_config_test",
    srcs = ["saved_model_config_test.cc"],
    data = [
        "//machina_serving/servables/machina/testdata:saved_model_half_plus_two_cpu_with_empty_saved_model_config",
        "//machina_serving/servables/machina/testdata:saved_model_half_plus_two_cpu_with_saved_model_config",
        "@org_machina//machina/cc/saved_model:saved_model_half_plus_two",
    ],
    deps = [
        ":saved_model_config",
        "//machina_serving/core/test_util:test_main",
        "//machina_serving/servables/machina:remote_op_config_rewriter_cc_proto",
        "//machina_serving/servables/machina:saved_model_config_cc_proto",
        "//machina_serving/servables/machina:saved_model_config_util",
        "//machina_serving/test_util",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
        "@local_xla//xla/tsl/lib/core:status_test_util",
        "@local_xla//xla/tsl/platform:status",
        "@org_machina//machina/cc/saved_model:constants",
        "@org_machina//machina/core:core_cpu",
        "@org_machina//machina/core:lib",
        "@org_machina//machina/core:protos_all_cc",
        "@org_machina//machina/core/grappler/optimizers/inference:batch_op_rewriter_proto_cc",
    ],
)

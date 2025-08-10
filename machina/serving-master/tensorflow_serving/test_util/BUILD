# Description: Tensorflow Serving test utils.

package(default_visibility = [
    "//machina_serving:internal",
])

licenses(["notice"])

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
)

cc_library(
    name = "test_util",
    testonly = 1,
    srcs = ["test_util.cc"],
    hdrs = ["test_util.h"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "@com_google_googletest//:gtest",
        "@com_google_protobuf//:protobuf",
        "@local_tsl//tsl/platform:protobuf",
        "@org_machina//machina/core:lib",
        "@org_machina//machina/core:session_options",
        "@org_machina//machina/core:test",
    ],
)

cc_binary(
    name = "grpc_client",
    srcs = ["grpc_client.cc"],
    deps = [
        "//machina_serving/apis:prediction_service_cc_proto",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/strings",
        "@org_machina//machina/core:lib",
        "@org_machina//machina/core/platform:env",
    ],
)

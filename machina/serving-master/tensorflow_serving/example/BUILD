# Description: Tensorflow Serving examples.

package(
    default_visibility = ["//machina_serving:internal"],
    features = ["no_layering_check"],
)

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

cc_binary(
    name = "resnet_client_cc",
    srcs = [
        "resnet_client.cc",
    ],
    deps = [
        "//machina_serving/apis:prediction_service_cc_proto",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_protobuf//:protobuf_lite",
        "@org_machina//machina/core:framework",
        "@org_machina//machina/core:lib",
        "@org_machina//machina/core/platform:jpeg",
    ],
)

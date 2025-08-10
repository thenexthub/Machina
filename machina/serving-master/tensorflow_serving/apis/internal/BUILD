# Internal implementation details of serving APIs.

load("//machina_serving:serving.bzl", "serving_proto_library")

package(features = ["-layering_check"])

licenses(["notice"])

serving_proto_library(
    name = "serialized_input_proto",
    srcs = ["serialized_input.proto"],
    visibility = [
        "//machina_serving:internal",
        "@org_machina//machina_ranking/google:__pkg__",
    ],
    deps = [
    ],
)

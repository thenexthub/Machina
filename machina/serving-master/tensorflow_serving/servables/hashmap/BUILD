# Description: Tensorflow Serving hashmap servable.

load("//machina_serving:serving.bzl", "serving_proto_library")

package(
    default_visibility = ["//machina_serving:internal"],
    features = ["-layering_check"],
)

licenses(["notice"])

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
            "g3doc/sitemap.md",
        ],
    ),
)

cc_library(
    name = "hashmap_source_adapter",
    srcs = ["hashmap_source_adapter.cc"],
    hdrs = ["hashmap_source_adapter.h"],
    visibility = ["//visibility:private"],
    deps = [
        ":hashmap_source_adapter_cc_proto",
        "//machina_serving/core:simple_loader",
        "//machina_serving/core:source_adapter",
        "//machina_serving/core:storage_path",
        "@org_machina//machina/core:lib",
        "@org_machina//machina/core:machina",
    ],
)

cc_test(
    name = "hashmap_source_adapter_test",
    size = "medium",
    srcs = ["hashmap_source_adapter_test.cc"],
    deps = [
        ":hashmap_source_adapter",
        ":hashmap_source_adapter_cc_proto",
        "//machina_serving/core:loader",
        "//machina_serving/core:servable_data",
        "//machina_serving/core/test_util:test_main",
        "//machina_serving/util:any_ptr",
        "@org_machina//machina/core:lib",
        "@org_machina//machina/core:test",
    ],
)

serving_proto_library(
    name = "hashmap_source_adapter_proto",
    srcs = ["hashmap_source_adapter.proto"],
)

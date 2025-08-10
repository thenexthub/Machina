# Description: Model Server test utility

# Placeholder: load py_library
load("//machina_serving:serving.bzl", "if_google", "serving_proto_library")

package(
    default_visibility = [
        "//machina_serving:internal",
    ],
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
        ],
    ),
)

cc_library(
    name = "mock_server_core",
    testonly = 1,
    hdrs = ["mock_server_core.h"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "//machina_serving/apis:logging_cc_proto",
        "//machina_serving/apis:model_cc_proto",
        "//machina_serving/config:model_server_config_cc_proto",
        "//machina_serving/config:platform_config_cc_proto",
        "//machina_serving/core:aspired_versions_manager",
        "//machina_serving/core:servable_handle",
        "//machina_serving/core:servable_id",
        "//machina_serving/core:servable_state",
        "//machina_serving/core:servable_state_monitor",
        "//machina_serving/core:server_request_logger",
        "//machina_serving/core/test_util:fake_loader_source_adapter",
        "//machina_serving/core/test_util:fake_loader_source_adapter_cc_proto",
        "//machina_serving/core/test_util:servable_handle_test_util",
        "//machina_serving/model_servers:server_core",
        "//machina_serving/servables/machina:mock_servable",
        "//machina_serving/servables/machina:servable",
        "//machina_serving/util:event_bus",
        "//machina_serving/util:unique_ptr_with_deps",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest",
        "@com_google_protobuf//:cc_wkt_protos",
        "@com_google_protobuf//:protobuf_lite",
        "@org_machina//machina/core:lib",
    ],
)

cc_library(
    name = "server_core_test_util",
    testonly = 1,
    srcs = ["server_core_test_util.cc"],
    hdrs = ["server_core_test_util.h"],
    data = [
        "//machina_serving/servables/machina/testdata:saved_model_half_plus_two_2_versions",
        "@org_machina//machina/cc/saved_model:saved_model_half_plus_two",
    ] + if_google([
        "//machina_serving/servables/machina/google/testdata:half_plus_two/00000123/export.data-00000-of-00001",
        "//machina_serving/servables/machina/google/testdata:half_plus_two/00000123/export.index",
        "//machina_serving/servables/machina/google/testdata:half_plus_two/00000123/export.meta",
        "//machina_serving/servables/machina/google/testdata:half_plus_two_2_versions/00000123/export.data-00000-of-00001",
        "//machina_serving/servables/machina/google/testdata:half_plus_two_2_versions/00000123/export.index",
        "//machina_serving/servables/machina/google/testdata:half_plus_two_2_versions/00000123/export.meta",
        "//machina_serving/servables/machina/google/testdata:half_plus_two_2_versions/00000124/export.data-00000-of-00001",
        "//machina_serving/servables/machina/google/testdata:half_plus_two_2_versions/00000124/export.index",
        "//machina_serving/servables/machina/google/testdata:half_plus_two_2_versions/00000124/export.meta",
    ]),
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "//machina_serving/core:availability_preserving_policy",
        "//machina_serving/core:servable_id",
        "//machina_serving/core/test_util:fake_loader_source_adapter",
        "//machina_serving/model_servers:model_platform_types",
        "//machina_serving/model_servers:platform_config_util",
        "//machina_serving/model_servers:server_core",
        "//machina_serving/servables/machina:saved_model_bundle_source_adapter_cc_proto",
        "//machina_serving/servables/machina:session_bundle_config_cc_proto",
        "//machina_serving/test_util",
        "@com_google_googletest//:gtest",
        "@org_machina//machina/core:lib",
        "@org_machina//machina/core:test",
    ],
)

cc_library(
    name = "storage_path_error_injecting_source_adapter",
    testonly = 1,
    srcs = ["storage_path_error_injecting_source_adapter.cc"],
    hdrs = ["storage_path_error_injecting_source_adapter.h"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        ":storage_path_error_injecting_source_adapter_cc_proto",
        "//machina_serving/core:source_adapter",
    ],
)

serving_proto_library(
    name = "storage_path_error_injecting_source_adapter_proto",
    testonly = 1,
    srcs = ["storage_path_error_injecting_source_adapter.proto"],
    visibility = [
        "//visibility:public",
    ],
)

py_library(
    name = "machina_model_server_test_base",
    srcs = ["machina_model_server_test_base.py"],
    srcs_version = "PY2AND3",
    tags = [
        "noasan",  # b/136281879
        "notsan",
    ],
    deps = [
        "//machina_serving/apis:predict_proto_py_pb2",
        "//machina_serving/apis:prediction_service_proto_py_pb2",
        "@org_machina//machina/core:protos_all_py",
        "@org_machina//machina/python/saved_model:signature_constants",
        "@six_archive//:six",
    ],
)

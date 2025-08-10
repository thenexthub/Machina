# Description: TensorFlow servables.

load("@org_machina//machina/core/platform:build_config.bzl", "tf_proto_library")
load("//machina_serving:serving.bzl", "if_google", "if_oss", "serving_proto_library", "serving_proto_library_py", "serving_machina_proto_dep")

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

serving_proto_library(
    name = "session_bundle_config_proto",
    srcs = ["session_bundle_config.proto"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "@com_google_protobuf//:cc_wkt_protos",
        serving_machina_proto_dep(
            "@org_machina//machina/core:protos_all",
        ),
    ],
)

serving_proto_library_py(
    name = "session_bundle_config_proto_py_pb2",
    srcs = ["session_bundle_config.proto"],
    proto_library = "session_bundle_config_proto",
    visibility = ["//visibility:private"],
    deps = [
        "@org_machina//machina/core:protos_all_py",
    ],
)

serving_proto_library_py(
    name = "session_bundle_config_py_pb2",
    proto_library = "session_bundle_config_proto",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "resource_estimator",
    hdrs = ["resource_estimator.h"],
    deps = ["//machina_serving/util:oss_or_google"] + if_oss([
        "//machina_serving/servables/machina/oss:resource_estimator",
    ]) + if_google(["//machina_serving/servables/machina/google:resource_estimator"]),
)

cc_library(
    name = "bundle_factory_util",
    srcs = ["bundle_factory_util.cc"],
    hdrs = ["bundle_factory_util.h"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        ":resource_estimator",
        ":serving_session",
        ":session_bundle_config_cc_proto",
        "//machina_serving/batching:batching_session",
        "//machina_serving/resources:resource_values",
        "//machina_serving/resources:resources_cc_proto",
        "//machina_serving/util:file_probing_env",
        "//machina_serving/util:proto_util",
        "@com_google_absl//absl/types:optional",
        "@com_google_protobuf//:cc_wkt_protos",
        "@org_machina//machina/core:core_cpu",
        "@org_machina//machina/core:lib",
        "@org_machina//machina/core:protos_all_cc",
        "@org_machina//machina/core/kernels/batching_util:batch_scheduler",
        "@org_machina//machina/core/kernels/batching_util:shared_batch_scheduler",
    ],
)

cc_test(
    name = "bundle_factory_util_test",
    size = "medium",
    srcs = ["bundle_factory_util_test.cc"],
    data = [
        "//machina_serving/session_bundle:session_bundle_half_plus_two",
        "@org_machina//machina/cc/saved_model:saved_model_half_plus_two",
    ],
    deps = [
        ":bundle_factory_test_util",
        ":bundle_factory_util",
        ":session_bundle_config_cc_proto",
        "//machina_serving/batching:batching_session",
        "//machina_serving/core/test_util:test_main",
        "//machina_serving/resources:resources_cc_proto",
        "//machina_serving/session_bundle:session_bundle_util",
        "//machina_serving/test_util",
        "//machina_serving/util/test_util:mock_file_probing_env",
        "@com_google_protobuf//:cc_wkt_protos",
        "@org_machina//machina/core:core_cpu",
        "@org_machina//machina/core:framework",
        "@org_machina//machina/core:lib",
        "@org_machina//machina/core:protos_all_cc",
        "@org_machina//machina/core:test",
        "@org_machina//machina/core/kernels/batching_util:shared_batch_scheduler",
    ],
)

cc_library(
    name = "bundle_factory_test_util",
    testonly = 1,
    srcs = ["bundle_factory_test_util.cc"],
    hdrs = ["bundle_factory_test_util.h"],
    deps = [
        "//machina_serving/resources:resource_values",
        "//machina_serving/resources:resources_cc_proto",
        "//machina_serving/test_util",
        "@org_machina//machina/cc/saved_model:constants",
        "@org_machina//machina/core:core_cpu",
        "@org_machina//machina/core:framework",
        "@org_machina//machina/core:lib",
        "@org_machina//machina/core:test",
        "@org_machina//machina/core:testlib",
    ],
)

cc_library(
    name = "bundle_factory_test",
    testonly = 1,
    srcs = ["bundle_factory_test.h"],
    deps = [
        ":bundle_factory_test_util",
        ":session_bundle_config_cc_proto",
        "//machina_serving/resources:resources_cc_proto",
        "//machina_serving/test_util",
        "@com_google_protobuf//:cc_wkt_protos",
        "@org_machina//machina/cc/saved_model:tag_constants",
        "@org_machina//machina/core:core_cpu",
        "@org_machina//machina/core:lib",
        "@org_machina//machina/core:test",
    ],
)

cc_library(
    name = "saved_model_bundle_factory",
    srcs = ["saved_model_bundle_factory.cc"],
    hdrs = ["saved_model_bundle_factory.h"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        ":bundle_factory_util",
        ":saved_model_config_cc_proto",
        ":saved_model_config_util",
        ":session_bundle_config_cc_proto",
        ":tflite_session_lib",
        "//machina_serving/batching:batching_session",
        "//machina_serving/core:loader",
        "//machina_serving/resources:resources_cc_proto",
        "//machina_serving/session_bundle:session_bundle_util",  # buildcleaner: keep
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@org_machina//machina/cc/saved_model:loader",
        "@org_machina//machina/cc/saved_model:tag_constants",
        "@org_machina//machina/core:core_cpu",
        "@org_machina//machina/core:lib",
        "@org_machina//machina/core:protos_all_cc",
        "@org_machina//machina/core/kernels/batching_util:shared_batch_scheduler",
    ],
)

cc_test(
    name = "saved_model_bundle_factory_test",
    size = "medium",
    srcs = ["saved_model_bundle_factory_test.cc"],
    data = [
        "//machina_serving/servables/machina/testdata:saved_model_half_plus_two_tflite",
        "//machina_serving/session_bundle:session_bundle_half_plus_two",
        "@org_machina//machina/cc/saved_model:saved_model_half_plus_two",
    ],
    deps = [
        ":bundle_factory_test",
        ":bundle_factory_test_util",
        ":saved_model_bundle_factory",
        ":saved_model_config_cc_proto",
        ":session_bundle_config_cc_proto",
        "//machina_serving/core/test_util:session_test_util",
        "//machina_serving/core/test_util:test_main",
        "@com_google_absl//absl/status",
        "@com_google_protobuf//:cc_wkt_protos",
        "@org_machina//machina/cc/saved_model:constants",
        "@org_machina//machina/cc/saved_model:loader",
        "@org_machina//machina/cc/saved_model:tag_constants",
        "@org_machina//machina/core:core_cpu",
        "@org_machina//machina/core:framework",
        "@org_machina//machina/core:lib",
        "@org_machina//machina/core:portable_machina_test_lib",
        "@org_machina//machina/core:protos_all_cc",
        "@org_machina//machina/core:test",
    ],
)

cc_library(
    name = "saved_model_bundle_source_adapter",
    srcs = ["saved_model_bundle_source_adapter.cc"],
    hdrs = ["saved_model_bundle_source_adapter.h"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        ":bundle_factory_util",
        ":machine_learning_metadata",
        ":saved_model_bundle_factory",
        ":saved_model_bundle_source_adapter_cc_proto",
        ":saved_model_warmup",
        "//machina_serving/core:loader",
        "//machina_serving/core:simple_loader",
        "//machina_serving/core:source_adapter",
        "//machina_serving/core:storage_path",
        "//machina_serving/resources:resource_util",
        "//machina_serving/resources:resource_values",
        "//machina_serving/resources:resources_cc_proto",
        "//machina_serving/servables/machina:file_acl",
        "@org_machina//machina/cc/saved_model:loader",
        "@org_machina//machina/core:lib",
    ],
    alwayslink = 1,
)

cc_test(
    name = "saved_model_bundle_source_adapter_test",
    size = "medium",
    srcs = ["saved_model_bundle_source_adapter_test.cc"],
    data = [
        "//machina_serving/servables/machina/testdata:saved_model_half_plus_two_mlmd",
        "//machina_serving/session_bundle:session_bundle_half_plus_two",
        "@org_machina//machina/cc/saved_model:saved_model_half_plus_two",
    ],
    # Link in all registered kernels.
    linkstatic = 1,
    deps = [
        ":bundle_factory_test_util",
        ":saved_model_bundle_source_adapter",
        ":saved_model_bundle_source_adapter_cc_proto",
        ":session_bundle_config_cc_proto",
        "//machina_serving/core:loader",
        "//machina_serving/core:servable_data",
        "//machina_serving/core/test_util:session_test_util",
        "//machina_serving/core/test_util:test_main",
        "//machina_serving/resources:resources_cc_proto",
        "//machina_serving/test_util",
        "//machina_serving/util:oss_or_google",
        "@com_google_protobuf//:cc_wkt_protos",
        "@org_machina//machina/cc/saved_model:loader",
        "@org_machina//machina/core:lib",
        "@org_machina//machina/core:test",
    ],
)

serving_proto_library(
    name = "saved_model_bundle_source_adapter_proto",
    srcs = ["saved_model_bundle_source_adapter.proto"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        ":session_bundle_config_proto",
    ],
)

serving_proto_library_py(
    name = "saved_model_bundle_source_adapter_py_pb2",
    proto_library = "saved_model_bundle_source_adapter_proto",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "simple_servers",
    srcs = ["simple_servers.cc"],
    hdrs = ["simple_servers.h"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        ":saved_model_bundle_source_adapter",
        ":saved_model_bundle_source_adapter_cc_proto",
        "//machina_serving/config:file_system_storage_path_source_cc_proto",
        "//machina_serving/core:aspired_versions_manager_builder",
        "//machina_serving/core:availability_preserving_policy",
        "//machina_serving/core:loader",
        "//machina_serving/core:manager",
        "//machina_serving/core:source",
        "//machina_serving/core:source_adapter",
        "//machina_serving/core:storage_path",
        "//machina_serving/core:target",
        "//machina_serving/sources/storage_path:file_system_storage_path_source",
        "@org_machina//machina/cc/saved_model:loader",
        "@org_machina//machina/core:lib",
    ],
)

cc_test(
    name = "simple_servers_test",
    srcs = ["simple_servers_test.cc"],
    data = ["@org_machina//machina/cc/saved_model:saved_model_half_plus_two"],
    # Link in all registered kernels.
    linkstatic = 1,
    deps = [
        ":simple_servers",
        "//machina_serving/core:servable_handle",
        "//machina_serving/core/test_util:test_main",
        "//machina_serving/test_util",
        "@org_machina//machina/cc/saved_model:loader",
        "@org_machina//machina/core:core_cpu",
        "@org_machina//machina/core:framework",
        "@org_machina//machina/core:lib",
        "@org_machina//machina/core:test",
        "@org_machina//machina/core:testlib",
    ],
)

cc_library(
    name = "serving_session",
    srcs = ["serving_session.cc"],
    hdrs = ["serving_session.h"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "@org_machina//machina/core:core_cpu",
        "@org_machina//machina/core:lib",
        "@org_machina//machina/core:protos_all_cc",
    ],
)

cc_library(
    name = "tflite_session_lib",
    srcs = [
        "tflite_interpreter_pool.cc",
        "tflite_session.cc",
    ],
    hdrs = [
        "tflite_interpreter_pool.h",
        "tflite_session.h",
    ],
    deps = [
        ":serving_session",
        "//machina_serving/batching:incremental_barrier",
        "//machina_serving/batching:threadsafe_status",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/functional:bind_front",
        "@com_google_absl//absl/synchronization",
        "@org_machina//machina/cc/saved_model:signature_constants",
        "@org_machina//machina/core:framework",
        "@org_machina//machina/core:lib",
        "@org_machina//machina/core:protos_all_cc",
        "@org_machina//machina/core:session_options",
        "@org_machina//machina/core/kernels/batching_util:basic_batch_scheduler",
        "@org_machina//machina/lite:external_cpu_backend_context",
        "@org_machina//machina/lite:framework",
        "@org_machina//machina/lite:string_util",
        "@org_machina//machina/lite:util",
        "@org_machina//machina/lite/c:common",
        "@org_machina//machina/lite/delegates/flex:delegate",
        "@org_machina//machina/lite/kernels:builtin_ops",
        "@org_machina//machina/lite/kernels:cpu_backend_context",
        "@org_machina//machina/lite/kernels/internal:tensor_utils",
        "@org_machina//machina/lite/kernels/parse_example",
        "@org_machina//machina/lite/profiling:profile_summarizer",
        "@org_machina//machina/lite/profiling:profile_summary_formatter",
        "@org_machina//machina/lite/profiling:profiler",
        "@org_machina//machina/lite/tools/signature:signature_def_util",
    ],
)

cc_test(
    name = "tflite_session_test",
    size = "small",
    srcs = ["tflite_session_test.cc"],
    data = [
        "//machina_serving/servables/machina/testdata:mobilenet_v1_quant_tflite",
        "//machina_serving/servables/machina/testdata:parse_example_tflite",
        "//machina_serving/servables/machina/testdata:saved_model_half_plus_two_tflite",
        "//machina_serving/servables/machina/testdata:saved_model_half_plus_two_tflite_with_sigdef",
    ],
    deps = [
        ":tflite_session_lib",
        "//machina_serving/core/test_util:test_main",
        "//machina_serving/test_util",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/functional:bind_front",
        "@flatbuffers",
        "@org_machina//machina/cc/saved_model:signature_constants",
        "@org_machina//machina/core:lib",
        "@org_machina//machina/core:protos_all_cc",
        "@org_machina//machina/core:test",
        "@org_machina//machina/core:testlib",
        "@org_machina//machina/lite:string_util",
        "@org_machina//machina/lite:util",
        "@org_machina//machina/lite:version",
        "@org_machina//machina/lite/tools/signature:signature_def_util",
    ],
)

cc_test(
    name = "tflite_interpreter_pool_test",
    size = "small",
    srcs = ["tflite_interpreter_pool_test.cc"],
    data = [
        "//machina_serving/servables/machina/testdata:mobilenet_v1_quant_tflite",
        "//machina_serving/servables/machina/testdata:parse_example_tflite",
        "//machina_serving/servables/machina/testdata:saved_model_half_plus_two_tflite",
        "//machina_serving/servables/machina/testdata:saved_model_half_plus_two_tflite_with_sigdef",
    ],
    deps = [
        ":tflite_session_lib",
        "//machina_serving/core/test_util:test_main",
        "//machina_serving/test_util",
        "@com_google_absl//absl/flags:flag",
        "@com_google_protobuf//:protobuf",
        "@flatbuffers",
        "@org_machina//machina/cc/saved_model:signature_constants",
        "@org_machina//machina/core:lib",
        "@org_machina//machina/core:protos_all_cc",
        "@org_machina//machina/core:test",
        "@org_machina//machina/core:testlib",
        "@org_machina//machina/lite:string_util",
        "@org_machina//machina/lite:util",
        "@org_machina//machina/lite:version",
        "@org_machina//machina/lite/tools/signature:signature_def_util",
    ],
)

cc_binary(
    name = "tflite_session",
    srcs = ["tflite_session_main.cc"],
    deps = [
        ":tflite_session_lib",
        "@org_machina//machina/core:lib",
    ],
)

cc_library(
    name = "predict_impl",
    srcs = ["predict_impl.cc"],
    hdrs = ["predict_impl.h"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        ":predict_util",
        ":thread_pool_factory",
        ":util",
        "//machina_serving/apis:predict_cc_proto",
        "//machina_serving/core:servable_handle",
        "//machina_serving/model_servers:server_core",
        "@com_google_absl//absl/strings",
        "@org_machina//machina/cc/saved_model:loader",
        "@org_machina//machina/core:lib",
        "@org_machina//machina/core:protos_all_cc",
    ],
)

cc_test(
    name = "predict_impl_test",
    size = "medium",
    srcs = ["predict_impl_test.cc"],
    data = [
        "//machina_serving/servables/machina/testdata:saved_model_counter/00000123/saved_model.pb",
        "//machina_serving/servables/machina/testdata:saved_model_counter/00000123/variables/variables.data-00000-of-00001",
        "//machina_serving/servables/machina/testdata:saved_model_counter/00000123/variables/variables.index",
        "@org_machina//machina/cc/saved_model:saved_model_half_plus_two",
    ],
    deps = [
        ":predict_impl",
        ":saved_model_bundle_source_adapter_cc_proto",
        ":session_bundle_config_cc_proto",
        "//machina_serving/core:availability_preserving_policy",
        "//machina_serving/core/test_util:test_main",
        "//machina_serving/model_servers:model_platform_types",
        "//machina_serving/model_servers:platform_config_util",
        "//machina_serving/model_servers:server_core",
        "//machina_serving/servables/machina/test_util:fake_thread_pool_factory",
        "//machina_serving/servables/machina/test_util:fake_thread_pool_factory_cc_proto",
        "//machina_serving/test_util",
        "@org_machina//machina/cc/saved_model:signature_constants",
        "@org_machina//machina/core:test",
    ],
)

cc_library(
    name = "predict_response_tensor_serialization_option",
    hdrs = ["predict_response_tensor_serialization_option.h"],
    visibility = [
        "//visibility:public",
    ],
)

cc_library(
    name = "predict_util",
    srcs = ["predict_util.cc"],
    hdrs = ["predict_util.h"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        ":predict_response_tensor_serialization_option",
        ":util",
        "//machina_serving/apis:predict_cc_proto",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@org_machina//machina/cc/saved_model:signature_constants",
        "@org_machina//machina/cc/saved_model:util",
        "@org_machina//machina/core:core_cpu",
        "@org_machina//machina/core:lib",
        "@org_machina//machina/core:protos_all_cc",
    ],
)

cc_test(
    name = "predict_util_test",
    size = "medium",
    srcs = ["predict_util_test.cc"],
    data = [
        "//machina_serving/servables/machina/testdata:bad_half_plus_two/00000123/export",
        "//machina_serving/servables/machina/testdata:bad_half_plus_two/00000123/export.meta",
        "//machina_serving/servables/machina/testdata:saved_model_counter/00000123/saved_model.pb",
        "//machina_serving/servables/machina/testdata:saved_model_counter/00000123/variables/variables.data-00000-of-00001",
        "//machina_serving/servables/machina/testdata:saved_model_counter/00000123/variables/variables.index",
        "@org_machina//machina/cc/saved_model:saved_model_half_plus_two",
    ] + if_google([
        "//machina_serving/servables/machina/google/testdata:half_plus_two/00000123/export.data-00000-of-00001",
        "//machina_serving/servables/machina/google/testdata:half_plus_two/00000123/export.index",
        "//machina_serving/servables/machina/google/testdata:half_plus_two/00000123/export.meta",
    ]),
    deps = [
        ":predict_util",
        ":saved_model_bundle_source_adapter_cc_proto",
        ":session_bundle_config_cc_proto",
        ":util",
        "//machina_serving/core:availability_preserving_policy",
        "//machina_serving/core/test_util:test_main",
        "//machina_serving/model_servers:model_platform_types",
        "//machina_serving/model_servers:platform_config_util",
        "//machina_serving/model_servers:server_core",
        "//machina_serving/test_util",
        "//machina_serving/util:oss_or_google",
        "@com_google_absl//absl/strings",
        "@org_machina//machina/cc/saved_model:loader",
        "@org_machina//machina/cc/saved_model:signature_constants",
        "@org_machina//machina/core:test",
    ],
)

cc_library(
    name = "get_model_metadata_impl",
    srcs = ["get_model_metadata_impl.cc"],
    hdrs = ["get_model_metadata_impl.h"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "//machina_serving/apis:get_model_metadata_cc_proto",
        "//machina_serving/core:servable_handle",
        "//machina_serving/model_servers:server_core",
        "@org_machina//machina/cc/saved_model:loader",
        "@org_machina//machina/core:lib",
    ],
)

cc_test(
    name = "get_model_metadata_impl_test",
    size = "medium",
    srcs = ["get_model_metadata_impl_test.cc"],
    data = [
        "@org_machina//machina/cc/saved_model:saved_model_half_plus_two",
    ] + if_google([
        "//machina_serving/servables/machina/google/testdata:half_plus_two/00000123/export.data-00000-of-00001",
        "//machina_serving/servables/machina/google/testdata:half_plus_two/00000123/export.index",
        "//machina_serving/servables/machina/google/testdata:half_plus_two/00000123/export.meta",
    ]),
    deps = [
        ":get_model_metadata_impl",
        ":saved_model_bundle_source_adapter_cc_proto",
        ":session_bundle_config_cc_proto",
        "//machina_serving/apis:model_cc_proto",
        "//machina_serving/config:model_server_config_cc_proto",
        "//machina_serving/config:platform_config_cc_proto",
        "//machina_serving/core:aspired_version_policy",
        "//machina_serving/core:availability_preserving_policy",
        "//machina_serving/core:servable_handle",
        "//machina_serving/core/test_util:test_main",
        "//machina_serving/model_servers:model_platform_types",
        "//machina_serving/model_servers:platform_config_util",
        "//machina_serving/test_util",
        "//machina_serving/util:oss_or_google",
        "@com_google_protobuf//:cc_wkt_protos",
        "@com_google_protobuf//:protobuf",
        "@org_machina//machina/cc/saved_model:loader",
        "@org_machina//machina/cc/saved_model:signature_constants",
        "@org_machina//machina/core:lib",
        "@org_machina//machina/core:protos_all_cc",
        "@org_machina//machina/core:test",
    ],
)

cc_library(
    name = "classifier",
    srcs = ["classifier.cc"],
    hdrs = ["classifier.h"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        ":util",
        "//machina_serving/apis:classification_cc_proto",
        "//machina_serving/apis:classifier",
        "//machina_serving/apis:input_cc_proto",
        "//machina_serving/apis:model_cc_proto",
        "@com_google_absl//absl/types:optional",
        "@org_machina//machina/cc/saved_model:loader_lite",
        "@org_machina//machina/cc/saved_model:signature_constants",
        "@org_machina//machina/core:framework",
        "@org_machina//machina/core:lib",
        "@org_machina//machina/core:protos_all_cc",
    ],
)

cc_test(
    name = "classifier_test",
    size = "medium",
    srcs = ["classifier_test.cc"],
    deps = [
        ":classifier",
        ":util",
        "//machina_serving/apis:classification_cc_proto",
        "//machina_serving/apis:input_cc_proto",
        "//machina_serving/apis:model_cc_proto",
        "//machina_serving/core/test_util:mock_session",
        "//machina_serving/core/test_util:test_main",
        "//machina_serving/test_util",
        "@com_google_absl//absl/types:optional",
        "@com_google_protobuf//:protobuf_lite",
        "@org_machina//machina/core:core_cpu",
        "@org_machina//machina/core:lib",
        "@org_machina//machina/core:protos_all_cc",
        "@org_machina//machina/core:test",
    ],
)

cc_library(
    name = "classification_service",
    srcs = ["classification_service.cc"],
    hdrs = ["classification_service.h"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        ":classifier",
        ":util",
        "//machina_serving/apis:classification_cc_proto",
        "//machina_serving/apis:classifier",
        "//machina_serving/core:servable_handle",
        "//machina_serving/model_servers:server_core",
        "@org_machina//machina/core:lib",
    ],
)

cc_test(
    name = "classification_service_test",
    srcs = ["classification_service_test.cc"],
    data = [
        "@org_machina//machina/cc/saved_model:saved_model_half_plus_two",
    ],
    deps = [
        ":classification_service",
        ":session_bundle_config_cc_proto",
        "//machina_serving/config:model_server_config_cc_proto",
        "//machina_serving/core:availability_preserving_policy",
        "//machina_serving/core/test_util:test_main",
        "//machina_serving/model_servers:model_platform_types",
        "//machina_serving/model_servers:platform_config_util",
        "//machina_serving/model_servers:server_core",
        "//machina_serving/test_util",
        "@org_machina//machina/core:lib",
        "@org_machina//machina/core:protos_all_cc",
        "@org_machina//machina/core:test",
    ],
)

cc_library(
    name = "regression_service",
    srcs = ["regression_service.cc"],
    hdrs = ["regression_service.h"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        ":regressor",
        ":util",
        "//machina_serving/apis:regression_cc_proto",
        "//machina_serving/apis:regressor",
        "//machina_serving/core:servable_handle",
        "//machina_serving/model_servers:server_core",
        "@org_machina//machina/core:lib",
    ],
)

cc_test(
    name = "regression_service_test",
    srcs = ["regression_service_test.cc"],
    data = [
        "@org_machina//machina/cc/saved_model:saved_model_half_plus_two",
    ],
    deps = [
        ":regression_service",
        ":session_bundle_config_cc_proto",
        "//machina_serving/config:model_server_config_cc_proto",
        "//machina_serving/core:availability_preserving_policy",
        "//machina_serving/core/test_util:test_main",
        "//machina_serving/model_servers:model_platform_types",
        "//machina_serving/model_servers:platform_config_util",
        "//machina_serving/model_servers:server_core",
        "//machina_serving/test_util",
        "@org_machina//machina/core:lib",
        "@org_machina//machina/core:protos_all_cc",
        "@org_machina//machina/core:test",
    ],
)

cc_library(
    name = "regressor",
    srcs = ["regressor.cc"],
    hdrs = ["regressor.h"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        ":util",
        "//machina_serving/apis:input_cc_proto",
        "//machina_serving/apis:model_cc_proto",
        "//machina_serving/apis:regression_cc_proto",
        "//machina_serving/apis:regressor",
        "@com_google_absl//absl/types:optional",
        "@org_machina//machina/cc/saved_model:loader_lite",
        "@org_machina//machina/cc/saved_model:signature_constants",
        "@org_machina//machina/core:framework",
        "@org_machina//machina/core:lib",
        "@org_machina//machina/core:protos_all_cc",
    ],
)

cc_test(
    name = "regressor_test",
    size = "medium",
    srcs = ["regressor_test.cc"],
    deps = [
        ":regressor",
        ":util",
        "//machina_serving/apis:input_cc_proto",
        "//machina_serving/apis:model_cc_proto",
        "//machina_serving/apis:regression_cc_proto",
        "//machina_serving/core/test_util:mock_session",
        "//machina_serving/core/test_util:test_main",
        "//machina_serving/test_util",
        "@com_google_absl//absl/types:optional",
        "@com_google_protobuf//:protobuf_lite",
        "@org_machina//machina/core:core_cpu",
        "@org_machina//machina/core:lib",
        "@org_machina//machina/core:protos_all_cc",
        "@org_machina//machina/core:test",
    ],
)

cc_library(
    name = "multi_inference",
    srcs = ["multi_inference.cc"],
    hdrs = ["multi_inference.h"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        ":classifier",
        ":regressor",
        ":util",
        "//machina_serving/apis:inference_cc_proto",
        "//machina_serving/apis:input_cc_proto",
        "//machina_serving/apis:model_cc_proto",
        "@com_google_absl//absl/types:optional",
        "@com_google_protobuf//:protobuf",
        "@org_machina//machina/cc/saved_model:signature_constants",
        "@org_machina//machina/core:framework",
        "@org_machina//machina/core:lib",
    ],
)

cc_test(
    name = "multi_inference_test",
    size = "medium",
    srcs = ["multi_inference_test.cc"],
    data = [
        "//machina_serving/servables/machina/testdata:saved_model_half_plus_two_tf2_cpu",
        "@org_machina//machina/cc/saved_model:saved_model_half_plus_two",
    ],
    deps = [
        ":multi_inference",
        ":session_bundle_config_cc_proto",
        ":util",
        "//machina_serving/apis:classification_cc_proto",
        "//machina_serving/apis:input_cc_proto",
        "//machina_serving/apis:regression_cc_proto",
        "//machina_serving/core:availability_preserving_policy",
        "//machina_serving/core/test_util:test_main",
        "//machina_serving/model_servers:model_platform_types",
        "//machina_serving/model_servers:platform_config_util",
        "//machina_serving/model_servers:server_core",
        "//machina_serving/test_util",
        "@org_machina//machina/cc/saved_model:loader",
        "@org_machina//machina/cc/saved_model:signature_constants",
        "@org_machina//machina/core:protos_all_cc",
        "@org_machina//machina/core:test",
    ],
)

cc_library(
    name = "multi_inference_helper",
    srcs = ["multi_inference_helper.cc"],
    hdrs = ["multi_inference_helper.h"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        ":multi_inference",
        "//machina_serving/apis:inference_cc_proto",
        "//machina_serving/apis:input_cc_proto",
        "//machina_serving/apis:model_cc_proto",
        "//machina_serving/model_servers:server_core",
        "@com_google_absl//absl/types:optional",
        "@com_google_protobuf//:protobuf",
        "@org_machina//machina/cc/saved_model:loader",
        "@org_machina//machina/core:framework",
        "@org_machina//machina/core:lib",
        "@org_machina//machina/core:protos_all_cc",
    ],
)

cc_test(
    name = "multi_inference_helper_test",
    size = "medium",
    srcs = ["multi_inference_helper_test.cc"],
    data = [
        "//machina_serving/servables/machina/testdata:saved_model_half_plus_two_tf2_cpu",
        "@org_machina//machina/cc/saved_model:saved_model_half_plus_two",
    ],
    deps = [
        ":multi_inference_helper",
        ":session_bundle_config_cc_proto",
        ":util",
        "//machina_serving/apis:classification_cc_proto",
        "//machina_serving/apis:input_cc_proto",
        "//machina_serving/apis:regression_cc_proto",
        "//machina_serving/core:availability_preserving_policy",
        "//machina_serving/core/test_util:test_main",
        "//machina_serving/model_servers:model_platform_types",
        "//machina_serving/model_servers:platform_config_util",
        "//machina_serving/model_servers:server_core",
        "//machina_serving/test_util",
        "@org_machina//machina/cc/saved_model:loader",
        "@org_machina//machina/cc/saved_model:signature_constants",
        "@org_machina//machina/core:protos_all_cc",
        "@org_machina//machina/core:test",
    ],
)

cc_library(
    name = "util",
    srcs = ["util.cc"],
    hdrs = ["util.h"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "//machina_serving/apis:input_cc_proto",
        "//machina_serving/apis:model_cc_proto",
        "//machina_serving/apis/internal:serialized_input_cc_proto",
        "//machina_serving/resources:resource_values",
        "//machina_serving/resources:resources_cc_proto",
        "//machina_serving/util:file_probing_env",
        "//machina_serving/util:threadpool_executor",
        "@com_google_absl//absl/types:optional",
        "@com_google_protobuf//:cc_wkt_protos",
        "@org_machina//machina/cc/saved_model:signature_constants",
        "@org_machina//machina/core:core_cpu",
        "@org_machina//machina/core:framework",
        "@org_machina//machina/core:lib",
        "@org_machina//machina/core:protos_all_cc",
    ],
)

cc_test(
    name = "util_test",
    size = "small",
    srcs = ["util_test.cc"],
    deps = [
        ":bundle_factory_test_util",
        ":util",
        "//machina_serving/core/test_util:test_main",
        "//machina_serving/test_util",
        "//machina_serving/util/test_util:mock_file_probing_env",
        "@com_google_protobuf//:cc_wkt_protos",
        "@org_machina//machina/cc/saved_model:signature_constants",
        "@org_machina//machina/core:framework",
        "@org_machina//machina/core:lib",
        "@org_machina//machina/core:protos_all_cc",
        "@org_machina//machina/core:test",
        "@org_machina//machina/core:testlib",
    ],
)

cc_library(
    name = "saved_model_warmup_util",
    srcs = ["saved_model_warmup_util.cc"],
    hdrs = ["saved_model_warmup_util.h"],
    deps = [
        ":session_bundle_config_cc_proto",
        "//machina_serving/apis:prediction_log_cc_proto",
        "//machina_serving/util:executor",
        "//machina_serving/util:threadpool_executor",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_protobuf//:cc_wkt_protos",
        "@org_machina//machina/cc/saved_model:constants",
        "@org_machina//machina/core:lib",
        "@org_machina//machina/core:protos_all_cc",
        "@org_machina//machina/core/kernels/batching_util:warmup",
    ],
)

cc_library(
    name = "saved_model_warmup_test_util",
    testonly = 1,
    srcs = ["saved_model_warmup_test_util.cc"],
    hdrs = ["saved_model_warmup_test_util.h"],
    deps = [
        ":session_bundle_config_cc_proto",
        "//machina_serving/apis:classification_cc_proto",
        "//machina_serving/apis:inference_cc_proto",
        "//machina_serving/apis:input_cc_proto",
        "//machina_serving/apis:model_cc_proto",
        "//machina_serving/apis:predict_cc_proto",
        "//machina_serving/apis:prediction_log_cc_proto",
        "//machina_serving/apis:regression_cc_proto",
        "@org_machina//machina/cc/saved_model:constants",
        "@org_machina//machina/cc/saved_model:signature_constants",
        "@org_machina//machina/core:lib",
        "@org_machina//machina/core:protos_all_cc",
    ],
)

cc_test(
    name = "saved_model_warmup_util_test",
    size = "small",
    srcs = ["saved_model_warmup_util_test.cc"],
    deps = [
        ":saved_model_warmup_test_util",
        ":saved_model_warmup_util",
        ":session_bundle_config_cc_proto",
        "//machina_serving/apis:classification_cc_proto",
        "//machina_serving/apis:inference_cc_proto",
        "//machina_serving/apis:input_cc_proto",
        "//machina_serving/apis:model_cc_proto",
        "//machina_serving/apis:predict_cc_proto",
        "//machina_serving/apis:prediction_log_cc_proto",
        "//machina_serving/apis:regression_cc_proto",
        "//machina_serving/core/test_util:test_main",  # buildcleaner: keep
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_protobuf//:cc_wkt_protos",
        "@local_xla//xla/tsl/lib/core:status_test_util",
        "@org_machina//machina/cc/saved_model:constants",
        "@org_machina//machina/cc/saved_model:loader",
        "@org_machina//machina/core:framework_lite",
        "@org_machina//machina/core:lib",
        "@org_machina//machina/core:protos_all_cc",
        "@org_machina//machina/core:test",
        "@org_machina//machina/core:tflite_portable_logging",
        "@org_machina//machina/core/example:example_protos_cc",
        "@org_machina//machina/core/kernels/batching_util:warmup",
        "@org_machina//machina/core/platform:errors",
        "@org_machina//machina/core/platform:path",
    ],
)

cc_library(
    name = "saved_model_warmup",
    srcs = ["saved_model_warmup.cc"],
    hdrs = ["saved_model_warmup.h"],
    deps = [
        ":classifier",
        ":multi_inference",
        ":predict_util",
        ":regressor",
        ":saved_model_warmup_util",
        ":session_bundle_config_cc_proto",
        ":util",
        "//machina_serving/apis:prediction_log_cc_proto",
        "@com_google_absl//absl/strings",
        "@com_google_protobuf//:cc_wkt_protos",
        "@org_machina//machina/cc/saved_model:constants",
        "@org_machina//machina/cc/saved_model:loader",
        "@org_machina//machina/core:core_cpu",
        "@org_machina//machina/core:lib",
        "@org_machina//machina/core:protos_all_cc",
    ],
)

cc_test(
    name = "saved_model_warmup_test",
    size = "small",
    srcs = ["saved_model_warmup_test.cc"],
    deps = [
        ":saved_model_warmup",
        ":saved_model_warmup_test_util",
        ":session_bundle_config_cc_proto",
        "//machina_serving/apis:classification_cc_proto",
        "//machina_serving/apis:inference_cc_proto",
        "//machina_serving/apis:input_cc_proto",
        "//machina_serving/apis:model_cc_proto",
        "//machina_serving/apis:predict_cc_proto",
        "//machina_serving/apis:prediction_log_cc_proto",
        "//machina_serving/apis:regression_cc_proto",
        "//machina_serving/core/test_util:mock_session",
        "//machina_serving/core/test_util:test_main",
        "@com_google_protobuf//:cc_wkt_protos",
        "@org_machina//machina/cc/saved_model:constants",
        "@org_machina//machina/cc/saved_model:signature_constants",
        "@org_machina//machina/core:lib",
        "@org_machina//machina/core:protos_all_cc",
        "@org_machina//machina/core:test",
    ],
)

cc_library(
    name = "machine_learning_metadata",
    srcs = ["machine_learning_metadata.cc"],
    hdrs = ["machine_learning_metadata.h"],
    deps = [
        "@org_machina//machina/cc/saved_model:constants",
        "@org_machina//machina/core:lib",
    ],
)

cc_test(
    name = "machine_learning_metadata_test",
    size = "small",
    srcs = ["machine_learning_metadata_test.cc"],
    data = [
        "//machina_serving/servables/machina/testdata:saved_model_half_plus_two_mlmd",
        "@org_machina//machina/cc/saved_model:saved_model_half_plus_two",
    ],
    deps = [
        ":bundle_factory_test_util",
        ":machine_learning_metadata",
        "//machina_serving/core/test_util:test_main",
        "//machina_serving/test_util",
        "@org_machina//machina/core:lib",
        "@org_machina//machina/core:testlib",
    ],
)

serving_proto_library(
    name = "thread_pool_factory_config_proto",
    srcs = ["thread_pool_factory_config.proto"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "@com_google_protobuf//:cc_wkt_protos",
    ],
)

cc_library(
    name = "thread_pool_factory",
    srcs = ["thread_pool_factory.cc"],
    hdrs = ["thread_pool_factory.h"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "//machina_serving/util:class_registration",
        "@org_machina//machina/core:lib",
    ],
)

cc_library(
    name = "servable",
    srcs = ["servable.cc"],
    hdrs = ["servable.h"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        ":run_options",
        "//machina_serving/apis:classification_cc_proto",
        "//machina_serving/apis:get_model_metadata_cc_proto",
        "//machina_serving/apis:inference_cc_proto",
        "//machina_serving/apis:predict_cc_proto",
        "//machina_serving/apis:regression_cc_proto",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@org_machina//machina/core:protos_all_cc",
    ],
)

cc_library(
    name = "mock_servable",
    testonly = True,
    hdrs = ["mock_servable.h"],
    visibility = [
        "//learning/gemini/deployment/disaggregation/wiz_py/cc_driver:__pkg__",
        "//machina_serving:internal",
    ],
    deps = [
        ":servable",
        "//machina_serving/apis:classification_cc_proto",
        "//machina_serving/apis:get_model_metadata_cc_proto",
        "//machina_serving/apis:inference_cc_proto",
        "//machina_serving/apis:predict_cc_proto",
        "//machina_serving/apis:regression_cc_proto",
        "//machina_serving/test_util",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_test(
    name = "thread_pool_factory_test",
    size = "small",
    srcs = ["thread_pool_factory_test.cc"],
    deps = [
        ":thread_pool_factory",
        "//machina_serving/core/test_util:test_main",
        "//machina_serving/test_util",
        "@org_machina//machina/core:lib",
        "@org_machina//machina/core/platform:threadpool_options",
    ],
)

cc_test(
    name = "servable_test",
    srcs = ["servable_test.cc"],
    deps = [
        ":servable",
        "//machina_serving/apis:predict_cc_proto",
        "//machina_serving/core/test_util:test_main",
        "//machina_serving/test_util",
        "@com_google_absl//absl/status",
    ],
)

cc_library(
    name = "tfrt_predict_util",
    srcs = ["tfrt_predict_util.cc"],
    hdrs = ["tfrt_predict_util.h"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        ":predict_response_tensor_serialization_option",
        ":predict_util",
        ":util",
        "//machina_serving/apis:predict_cc_proto",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@local_tsl//tsl/platform:error_logging",
        "@org_machina//machina/cc/saved_model:signature_constants",
        "@org_machina//machina/cc/saved_model:util",
        "@org_machina//machina/core:lib",
        "@org_machina//machina/core:protos_all_cc",
        "@org_machina//machina/core/platform:tracing",
        "@org_machina//machina/core/tfrt/runtime:tf_threadpool_concurrent_work_queue",
        "@org_machina//machina/core/tfrt/saved_model:saved_model_cpu",
    ],
)

cc_test(
    name = "tfrt_predict_util_test",
    srcs = ["tfrt_predict_util_test.cc"],
    data = [
        "//machina_serving/servables/machina/testdata:saved_model_half_plus_two_tf2_cpu",
    ],
    deps = [
        ":saved_model_bundle_source_adapter_cc_proto",
        ":servable",
        ":session_bundle_config_cc_proto",
        ":tfrt_predict_util",
        ":tfrt_saved_model_source_adapter",
        ":tfrt_saved_model_source_adapter_cc_proto",
        "//machina_serving/core:availability_preserving_policy",
        "//machina_serving/core:servable_handle",
        "//machina_serving/core/test_util:test_main",
        "//machina_serving/model_servers:model_platform_types",
        "//machina_serving/model_servers:platform_config_util",
        "//machina_serving/model_servers:server_core",
        "//machina_serving/servables/machina/test_util:mock_tfrt_saved_model",
        "//machina_serving/test_util",
        "//machina_serving/util:oss_or_google",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@org_machina//machina/cc/saved_model:loader_lite",
        "@org_machina//machina/cc/saved_model:signature_constants",
        "@org_machina//machina/core:lib",
        "@org_machina//machina/core:protos_all_cc",
        "@org_machina//machina/core:test",
        "@org_machina//machina/core/tfrt/saved_model:saved_model_cpu",
        "@org_machina//machina/core/tfrt/utils:tensor_util",
    ],
)

cc_library(
    name = "tfrt_saved_model_factory",
    srcs = ["tfrt_saved_model_factory.cc"],
    hdrs = ["tfrt_saved_model_factory.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":bundle_factory_util",
        ":machine_learning_metadata",
        ":saved_model_config",
        ":saved_model_config_util",
        ":servable",
        ":session_bundle_config_cc_proto",
        ":tfrt_saved_model_source_adapter_cc_proto",
        ":tfrt_saved_model_warmup",
        ":tfrt_servable",
        ":thread_pool_factory",
        ":thread_pool_factory_config_cc_proto",
        "//machina_serving/batching:tfrt_saved_model_with_batching",
        "//machina_serving/core:loader",
        "//machina_serving/resources:resource_values",
        "//machina_serving/resources:resources_cc_proto",
        "//machina_serving/session_bundle:graph_rewriter",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:optional",
        "@com_google_protobuf//:cc_wkt_protos",
        "@org_machina//machina/cc/saved_model:reader",
        "@org_machina//machina/cc/saved_model:tag_constants",
        "@org_machina//machina/compiler/mlir/tfrt:tfrt_compile_options",
        "@org_machina//machina/core:core_cpu",
        "@org_machina//machina/core:lib",
        "@org_machina//machina/core:protos_all_cc",
        "@org_machina//machina/core/kernels/batching_util:batch_scheduler_hdrs",
        "@org_machina//machina/core/kernels/batching_util:shared_batch_scheduler_hdrs",
        "@org_machina//machina/core/tfrt/graph_executor:graph_execution_options",
        "@org_machina//machina/core/tfrt/runtime",
        "@org_machina//machina/core/tfrt/saved_model:saved_model_cpu",
    ],
)

cc_test(
    name = "tfrt_saved_model_factory_test",
    size = "medium",
    srcs = ["tfrt_saved_model_factory_test.cc"],
    data = [
        "//machina_serving/servables/machina/testdata:saved_model_half_plus_two_cpu",
        "//machina_serving/servables/machina/testdata:saved_model_half_plus_two_cpu_with_saved_model_config",
    ],
    deps = [
        "//machina_serving/core/test_util:session_test_util",
        "//machina_serving/core/test_util:test_main",
        "//machina_serving/servables/machina:bundle_factory_test_util",
        "//machina_serving/servables/machina:tfrt_saved_model_factory",
        "//machina_serving/session_bundle:graph_rewriter",
        "//machina_serving/test_util",
        "@com_google_protobuf//:cc_wkt_protos",
        "@org_machina//machina/cc/saved_model:constants",
        "@org_machina//machina/cc/saved_model:tag_constants",
        "@org_machina//machina/core:protos_all_cc",
        "@org_machina//machina/core:test",
        "@org_machina//machina/core:testlib",
        "@org_machina//machina/core/tfrt/utils:tensor_util",
    ],
)

serving_proto_library(
    name = "tfrt_saved_model_source_adapter_proto",
    srcs = ["tfrt_saved_model_source_adapter.proto"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        ":session_bundle_config_proto",
        serving_machina_proto_dep(
            "@org_machina//machina/core:protos_all",
        ),
    ],
)

cc_library(
    name = "tfrt_saved_model_source_adapter",
    srcs = ["tfrt_saved_model_source_adapter.cc"],
    hdrs = ["tfrt_saved_model_source_adapter.h"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        ":bundle_factory_util",
        ":file_acl",
        ":machine_learning_metadata",
        ":servable",
        ":tfrt_saved_model_factory",
        ":tfrt_saved_model_source_adapter_cc_proto",
        ":tfrt_servable",
        "//machina_serving/core:loader",
        "//machina_serving/core:simple_loader",
        "//machina_serving/core:source_adapter",
        "//machina_serving/core:storage_path",
        "//machina_serving/resources:resource_util",
        "//machina_serving/resources:resource_values",
        "//machina_serving/resources:resources_cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@org_machina//machina/cc/saved_model:loader",
        "@org_machina//machina/core:lib",
    ],
    alwayslink = 1,
)

cc_test(
    name = "tfrt_saved_model_source_adapter_test",
    srcs = ["tfrt_saved_model_source_adapter_test.cc"],
    data = [
        "//machina_serving/servables/machina/testdata:saved_model_half_plus_two_cpu",
        "//machina_serving/servables/machina/testdata:saved_model_half_plus_two_mlmd",
    ],
    deps = [
        ":servable",
        ":tfrt_saved_model_source_adapter",
        ":tfrt_saved_model_source_adapter_cc_proto",
        ":tfrt_servable",
        "//machina_serving/core:loader",
        "//machina_serving/core/test_util:test_main",
        "//machina_serving/resources:resource_util",
        "//machina_serving/resources:resource_values",
        "//machina_serving/resources:resources_cc_proto",
        "//machina_serving/test_util",
        "@com_google_protobuf//:cc_wkt_protos",
        "@org_machina//machina/core:lib",
        "@org_machina//machina/core:test",
        "@org_machina//machina/core:testlib",
        "@org_machina//machina/core/tfrt/saved_model:saved_model_cpu",
        "@org_machina//machina/core/tfrt/utils:tensor_util",
    ],
)

cc_library(
    name = "tfrt_saved_model_warmup",
    srcs = ["tfrt_saved_model_warmup.cc"],
    hdrs = ["tfrt_saved_model_warmup.h"],
    visibility = [
        "//visibility:public",
    ] + if_google(["//learning/infra/mira:__subpackages__"]),
    deps = [
        ":saved_model_warmup_util",
        ":session_bundle_config_cc_proto",
        ":tfrt_classifier",
        ":tfrt_multi_inference",
        ":tfrt_predict_util",
        ":tfrt_regressor",
        ":util",
        "//machina_serving/apis:classification_cc_proto",
        "//machina_serving/apis:inference_cc_proto",
        "//machina_serving/apis:predict_cc_proto",
        "//machina_serving/apis:prediction_log_cc_proto",
        "//machina_serving/apis:regression_cc_proto",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_protobuf//:cc_wkt_protos",
        "@org_machina//machina/cc/saved_model:constants",
        "@org_machina//machina/cc/saved_model:loader",
        "@org_machina//machina/core:framework",
        "@org_machina//machina/core:lib",
        "@org_machina//machina/core:protos_all_cc",
        "@org_machina//machina/core/tfrt/saved_model:saved_model_cpu",
    ],
)

cc_test(
    name = "tfrt_saved_model_warmup_test",
    size = "small",
    srcs = ["tfrt_saved_model_warmup_test.cc"],
    deps = [
        ":saved_model_warmup_test_util",
        ":tfrt_saved_model_warmup",
        "//machina_serving/apis:classification_cc_proto",
        "//machina_serving/apis:inference_cc_proto",
        "//machina_serving/apis:input_cc_proto",
        "//machina_serving/apis:model_cc_proto",
        "//machina_serving/apis:predict_cc_proto",
        "//machina_serving/apis:prediction_log_cc_proto",
        "//machina_serving/apis:regression_cc_proto",
        "//machina_serving/core/test_util:test_main",
        "//machina_serving/servables/machina/test_util:mock_tfrt_saved_model",
        "@com_google_protobuf//:cc_wkt_protos",
        "@local_tsl//tsl/platform:path",
        "@org_machina//machina/cc/saved_model:constants",
        "@org_machina//machina/cc/saved_model:signature_constants",
        "@org_machina//machina/core:test",
        "@org_machina//machina/core/tfrt/graph_executor:graph_execution_options",
        "@org_machina//machina/core/tfrt/saved_model:saved_model_cpu",
        "@org_machina//machina/core/tfrt/utils:tensor_util",
    ],
)

cc_library(
    name = "tfrt_regression_service",
    srcs = ["tfrt_regression_service.cc"],
    hdrs = ["tfrt_regression_service.h"],
    visibility = ["//visibility:private"],
    deps = [
        ":servable",
        ":util",
        "//machina_serving/apis:regression_cc_proto",
        "//machina_serving/apis:regressor",
        "//machina_serving/core:servable_handle",
        "//machina_serving/model_servers:server_core",
        "@org_machina//machina/core:lib",
        "@org_machina//machina/core:protos_all_cc",
        "@org_machina//machina/core/platform:tracing",
    ],
)

cc_library(
    name = "tfrt_regressor",
    srcs = ["tfrt_regressor.cc"],
    hdrs = ["tfrt_regressor.h"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        ":util",
        "//machina_serving/apis:input_cc_proto",
        "//machina_serving/apis:model_cc_proto",
        "//machina_serving/apis:regression_cc_proto",
        "//machina_serving/apis:regressor",
        "@com_google_absl//absl/types:optional",
        "@local_tsl//tsl/platform:error_logging",
        "@org_machina//machina/cc/saved_model:loader_lite",
        "@org_machina//machina/cc/saved_model:signature_constants",
        "@org_machina//machina/core:framework",
        "@org_machina//machina/core:lib",
        "@org_machina//machina/core:protos_all_cc",
        "@org_machina//machina/core/platform:tracing",
        "@org_machina//machina/core/tfrt/saved_model:saved_model_cpu",
        "@org_machina//machina/core/tfrt/utils:tensor_util",
    ],
)

cc_test(
    name = "tfrt_regression_service_test",
    srcs = ["tfrt_regression_service_test.cc"],
    data = [
        "//machina_serving/servables/machina/testdata:saved_model_half_plus_two_cpu",
    ],
    deps = [
        ":session_bundle_config_cc_proto",
        ":tfrt_regression_service",
        ":tfrt_saved_model_source_adapter",
        ":tfrt_saved_model_source_adapter_cc_proto",
        "//machina_serving/config:model_server_config_cc_proto",
        "//machina_serving/core:availability_preserving_policy",
        "//machina_serving/core/test_util:test_main",
        "//machina_serving/model_servers:model_platform_types",
        "//machina_serving/model_servers:platform_config_util",
        "//machina_serving/model_servers:server_core",
        "//machina_serving/test_util",
        "@org_machina//machina/core:lib",
        "@org_machina//machina/core:protos_all_cc",
        "@org_machina//machina/core:test",
        "@org_machina//machina/core/tfrt/saved_model:saved_model_cpu",
    ],
)

cc_test(
    name = "tfrt_regressor_test",
    srcs = ["tfrt_regressor_test.cc"],
    data = [
        "//machina_serving/servables/machina/testdata:saved_model_half_plus_two_cpu",
    ],
    shard_count = 3,
    deps = [
        ":session_bundle_config_cc_proto",
        ":tfrt_regressor",
        ":tfrt_saved_model_source_adapter",
        ":tfrt_saved_model_source_adapter_cc_proto",
        ":tfrt_servable",
        "//machina_serving/apis:input_cc_proto",
        "//machina_serving/apis:model_cc_proto",
        "//machina_serving/apis:regression_cc_proto",
        "//machina_serving/config:model_server_config_cc_proto",
        "//machina_serving/core:availability_preserving_policy",
        "//machina_serving/core/test_util:test_main",
        "//machina_serving/model_servers:model_platform_types",
        "//machina_serving/model_servers:platform_config_util",
        "//machina_serving/model_servers:server_core",
        "//machina_serving/servables/machina/test_util:mock_tfrt_saved_model",
        "//machina_serving/test_util",
        "@com_google_protobuf//:protobuf_lite",
        "@org_machina//machina/cc/saved_model:signature_constants",
        "@org_machina//machina/core:core_cpu",
        "@org_machina//machina/core:lib",
        "@org_machina//machina/core:protos_all_cc",
        "@org_machina//machina/core:test",
        "@org_machina//machina/core/tfrt/utils:tensor_util",
    ],
)

cc_library(
    name = "tfrt_classifier",
    srcs = ["tfrt_classifier.cc"],
    hdrs = ["tfrt_classifier.h"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        ":util",
        "//machina_serving/apis:classification_cc_proto",
        "//machina_serving/apis:classifier",
        "//machina_serving/apis:input_cc_proto",
        "//machina_serving/apis:model_cc_proto",
        "@com_google_absl//absl/types:optional",
        "@local_tsl//tsl/platform:error_logging",
        "@org_machina//machina/cc/saved_model:loader_lite",
        "@org_machina//machina/cc/saved_model:signature_constants",
        "@org_machina//machina/core:framework",
        "@org_machina//machina/core:lib",
        "@org_machina//machina/core:protos_all_cc",
        "@org_machina//machina/core/platform:tracing",
        "@org_machina//machina/core/tfrt/saved_model:saved_model_cpu",
        "@org_machina//machina/core/tfrt/utils:tensor_util",
    ],
)

cc_test(
    name = "tfrt_classifier_test",
    size = "large",
    srcs = ["tfrt_classifier_test.cc"],
    data = [
        "//machina_serving/servables/machina/testdata:saved_model_half_plus_two_cpu",
    ],
    deps = [
        ":servable",
        ":session_bundle_config_cc_proto",
        ":tfrt_classifier",
        ":tfrt_saved_model_source_adapter",
        ":tfrt_saved_model_source_adapter_cc_proto",
        ":tfrt_servable",
        "//machina_serving/apis:classification_cc_proto",
        "//machina_serving/apis:input_cc_proto",
        "//machina_serving/apis:model_cc_proto",
        "//machina_serving/config:model_server_config_cc_proto",
        "//machina_serving/core:availability_preserving_policy",
        "//machina_serving/core:servable_handle",
        "//machina_serving/core/test_util:test_main",
        "//machina_serving/model_servers:model_platform_types",
        "//machina_serving/model_servers:platform_config_util",
        "//machina_serving/model_servers:server_core",
        "//machina_serving/servables/machina/test_util:mock_tfrt_saved_model",
        "//machina_serving/test_util",
        "@com_google_protobuf//:protobuf_lite",
        "@org_machina//machina/cc/saved_model:signature_constants",
        "@org_machina//machina/core:core_cpu",
        "@org_machina//machina/core:lib",
        "@org_machina//machina/core:protos_all_cc",
        "@org_machina//machina/core:test",
        "@org_machina//machina/core/tfrt/utils:tensor_util",
    ],
)

cc_library(
    name = "tfrt_classification_service",
    srcs = ["tfrt_classification_service.cc"],
    hdrs = ["tfrt_classification_service.h"],
    visibility = ["//visibility:private"],
    deps = [
        ":servable",
        ":tfrt_classifier",
        ":util",
        "//machina_serving/apis:classification_cc_proto",
        "//machina_serving/apis:classifier",
        "//machina_serving/core:servable_handle",
        "//machina_serving/model_servers:server_core",
        "@org_machina//machina/core:lib",
        "@org_machina//machina/core:protos_all_cc",
        "@org_machina//machina/core/platform:tracing",
    ],
)

cc_test(
    name = "tfrt_classification_service_test",
    srcs = ["tfrt_classification_service_test.cc"],
    data = [
        "//machina_serving/servables/machina/testdata:saved_model_half_plus_two_cpu",
    ],
    deps = [
        ":session_bundle_config_cc_proto",
        ":tfrt_classification_service",
        ":tfrt_saved_model_source_adapter",
        ":tfrt_saved_model_source_adapter_cc_proto",
        "//machina_serving/config:model_server_config_cc_proto",
        "//machina_serving/core:availability_preserving_policy",
        "//machina_serving/core/test_util:test_main",
        "//machina_serving/model_servers:model_platform_types",
        "//machina_serving/model_servers:platform_config_util",
        "//machina_serving/model_servers:server_core",
        "//machina_serving/test_util",
        "@org_machina//machina/core:lib",
        "@org_machina//machina/core:protos_all_cc",
        "@org_machina//machina/core:test",
        "@org_machina//machina/core/tfrt/runtime",
    ],
)

cc_library(
    name = "tfrt_multi_inference",
    srcs = ["tfrt_multi_inference.cc"],
    hdrs = ["tfrt_multi_inference.h"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        ":tfrt_classifier",
        ":tfrt_regressor",
        ":util",
        "//machina_serving/apis:inference_cc_proto",
        "//machina_serving/apis:input_cc_proto",
        "//machina_serving/apis:model_cc_proto",
        "//machina_serving/model_servers:server_core",
        "@com_google_absl//absl/types:optional",
        "@local_tsl//tsl/platform:error_logging",
        "@org_machina//machina/cc/saved_model:loader_lite",
        "@org_machina//machina/cc/saved_model:signature_constants",
        "@org_machina//machina/core:lib",
        "@org_machina//machina/core/tfrt/saved_model:saved_model_cpu",
        "@org_machina//machina/core/tfrt/utils:tensor_util",
    ],
)

cc_test(
    name = "tfrt_multi_inference_test",
    size = "medium",
    srcs = ["tfrt_multi_inference_test.cc"],
    data = [
        "//machina_serving/servables/machina/testdata:saved_model_half_plus_two_cpu",
    ],
    deps = [
        ":servable",
        ":session_bundle_config_cc_proto",
        ":tfrt_multi_inference",
        ":tfrt_saved_model_source_adapter",
        ":tfrt_saved_model_source_adapter_cc_proto",
        ":tfrt_servable",
        "//machina_serving/apis:classification_cc_proto",
        "//machina_serving/apis:input_cc_proto",
        "//machina_serving/apis:regression_cc_proto",
        "//machina_serving/core:availability_preserving_policy",
        "//machina_serving/core:servable_handle",
        "//machina_serving/core/test_util:test_main",
        "//machina_serving/model_servers:model_platform_types",
        "//machina_serving/model_servers:platform_config_util",
        "//machina_serving/model_servers:server_core",
        "//machina_serving/test_util",
        "@com_google_absl//absl/status",
        "@org_machina//machina/cc/saved_model:loader",
        "@org_machina//machina/cc/saved_model:signature_constants",
        "@org_machina//machina/core:protos_all_cc",
        "@org_machina//machina/core:test",
        "@org_machina//machina/core/tfrt/saved_model:saved_model_cpu",
    ],
)

cc_library(
    name = "tfrt_get_model_metadata_impl",
    srcs = ["tfrt_get_model_metadata_impl.cc"],
    hdrs = ["tfrt_get_model_metadata_impl.h"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        ":servable",
        "//machina_serving/apis:get_model_metadata_cc_proto",
        "//machina_serving/core:servable_handle",
        "//machina_serving/model_servers:server_core",
        "@org_machina//machina/core:lib",
    ],
)

cc_library(
    name = "tfrt_servable",
    srcs = ["tfrt_servable.cc"],
    hdrs = ["tfrt_servable.h"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        ":predict_response_tensor_serialization_option",
        ":saved_model_config_cc_proto",
        ":saved_model_config_util",
        ":servable",
        ":tfrt_classifier",
        ":tfrt_multi_inference",
        ":tfrt_predict_util",
        ":tfrt_regressor",
        ":tfrt_saved_model_source_adapter_cc_proto",
        ":thread_pool_factory",
        "//machina_serving/apis:classification_cc_proto",
        "//machina_serving/apis:get_model_metadata_cc_proto",
        "//machina_serving/apis:inference_cc_proto",
        "//machina_serving/apis:predict_cc_proto",
        "//machina_serving/apis:regression_cc_proto",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@local_xla//xla/tsl/platform:errors",
        "@local_xla//xla/tsl/platform:statusor",
        "@local_xla//xla/tsl/platform:threadpool_options",
        "@org_machina//machina/cc/saved_model:signature_constants",
        "@org_machina//machina/core:framework",
        "@org_machina//machina/core:protos_all_cc",
        "@org_machina//machina/core/platform:tracing",
        "@org_machina//machina/core/tfrt/saved_model:saved_model_cpu",
    ],
)

cc_test(
    name = "tfrt_get_model_metadata_impl_test",
    size = "medium",
    srcs = ["tfrt_get_model_metadata_impl_test.cc"],
    data = [
        "@org_machina//machina/cc/saved_model:saved_model_half_plus_two",
    ],
    deps = [
        ":session_bundle_config_cc_proto",
        ":tfrt_get_model_metadata_impl",
        ":tfrt_saved_model_source_adapter",
        ":tfrt_saved_model_source_adapter_cc_proto",
        ":tfrt_servable",
        "//machina_serving/apis:model_cc_proto",
        "//machina_serving/config:model_server_config_cc_proto",
        "//machina_serving/config:platform_config_cc_proto",
        "//machina_serving/core:aspired_version_policy",
        "//machina_serving/core:availability_preserving_policy",
        "//machina_serving/core:servable_handle",
        "//machina_serving/core/test_util:test_main",
        "//machina_serving/model_servers:model_platform_types",
        "//machina_serving/model_servers:platform_config_util",
        "//machina_serving/test_util",
        "@com_google_protobuf//:cc_wkt_protos",
        "@com_google_protobuf//:protobuf",
        "@local_xla//xla/tsl/lib/core:status_test_util",
        "@org_machina//machina/cc/saved_model:loader",
        "@org_machina//machina/cc/saved_model:signature_constants",
        "@org_machina//machina/core:lib",
        "@org_machina//machina/core:protos_all_cc",
        "@org_machina//machina/core:test",
    ] + if_google(["@com_google_absl//absl/strings:string_view"]),
)

serving_proto_library(
    name = "remote_op_config_rewriter_proto",
    srcs = ["remote_op_config_rewriter.proto"],
    visibility = ["//visibility:public"],
)

serving_proto_library_py(
    name = "remote_op_config_rewriter_proto_py_pb2",
    srcs = ["remote_op_config_rewriter.proto"],
    proto_library = "remote_op_config_rewriter_proto",
    visibility = ["//visibility:private"],
    deps = [],
)

# To generate *_genproto and *_cc_genproto of TF protos for Tensorflow serving target to use.
tf_proto_library(
    name = "tf_protos_deps",
    srcs = [],
    create_go_proto = False,
    create_java_proto = False,
    create_kotlin_proto = False,
    make_default_target_header_only = True,
    protodeps = [
        "@org_machina//machina/core/grappler/optimizers/inference:batch_op_rewriter_proto",
        "@org_machina//machina/core/tfrt/graph_executor:config_proto",
    ],
)

serving_proto_library(
    name = "saved_model_config_proto",
    srcs = ["saved_model_config.proto"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        ":remote_op_config_rewriter_proto",
        ":session_bundle_config_proto",
        serving_machina_proto_dep(":tf_protos_deps"),
        serving_machina_proto_dep("@org_machina//machina/core:protos_all"),
    ],
)

serving_proto_library_py(
    name = "saved_model_config_proto_py_pb2",
    srcs = ["saved_model_config.proto"],
    proto_library = "saved_model_config_proto",
    visibility = ["//visibility:public"],
    deps = [
        ":tf_protos_deps_py",
        "//machina_serving/servables/machina:remote_op_config_rewriter_proto_py_pb2",
        "//machina_serving/servables/machina:session_bundle_config_proto_py_pb2",
        "@org_machina//machina/core:protos_all_py",
    ],
)

cc_library(
    name = "saved_model_config_util",
    srcs = ["saved_model_config_util.cc"],
    hdrs = ["saved_model_config_util.h"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        ":remote_op_config_rewriter_cc_proto",
        ":saved_model_config_cc_proto",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@local_xla//xla/tsl/platform:types",
        "@org_machina//machina/cc/saved_model:constants",
        "@org_machina//machina/core:lib",
        "@org_machina//machina/core:protos_all_cc",
        "@org_machina//machina/core/grappler/optimizers/inference:batch_op_rewriter_proto_cc",
    ],
)

cc_test(
    name = "saved_model_config_util_test",
    srcs = ["saved_model_config_util_test.cc"],
    data = [
        "//machina_serving/servables/machina/testdata:saved_model_half_plus_two_cpu",
        "//machina_serving/servables/machina/testdata:saved_model_half_plus_two_cpu_with_empty_saved_model_config",
        "//machina_serving/servables/machina/testdata:saved_model_half_plus_two_cpu_with_saved_model_config",
    ],
    deps = [
        ":saved_model_config_cc_proto",
        ":saved_model_config_util",
        "//machina_serving/core/test_util:test_main",
        "//machina_serving/servables/machina:remote_op_config_rewriter_cc_proto",
        "//machina_serving/test_util",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
        "@com_google_protobuf//:protobuf",
        "@local_xla//xla/tsl/lib/core:status_test_util",
        "@org_machina//machina/core:protos_all_cc",
        "@org_machina//machina/core/grappler/optimizers/inference:batch_op_rewriter_proto_cc",
    ],
)

cc_library(
    name = "saved_model_config",
    srcs = ["saved_model_config.cc"],
    hdrs = ["saved_model_config.h"],
    visibility = ["//visibility:private"],
    deps = [
        ":saved_model_config_cc_proto",
        ":saved_model_config_util",
        "@com_google_absl//absl/status:statusor",
        "@org_machina//machina/core:lib",
        "@org_machina//machina/core:protos_all_cc",
        "@org_machina//machina/core/tfrt/graph_executor:config",
    ],
)

cc_test(
    name = "saved_model_config_test",
    srcs = ["saved_model_config_test.cc"],
    data = [
        "//machina_serving/servables/machina/testdata:saved_model_half_plus_two_cpu",
        "//machina_serving/servables/machina/testdata:saved_model_half_plus_two_cpu_with_empty_saved_model_config",
        "//machina_serving/servables/machina/testdata:saved_model_half_plus_two_cpu_with_saved_model_config",
    ],
    deps = [
        ":saved_model_config",
        ":saved_model_config_cc_proto",
        ":saved_model_config_util",
        "//machina_serving/core/test_util:test_main",
        "//machina_serving/servables/machina:remote_op_config_rewriter_cc_proto",
        "//machina_serving/test_util",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
        "@local_xla//xla/tsl/lib/core:status_test_util",
        "@local_xla//xla/tsl/platform:status",
        "@org_machina//machina/cc/saved_model:constants",
        "@org_machina//machina/core:lib",
        "@org_machina//machina/core:protos_all_cc",
        "@org_machina//machina/core/grappler/optimizers/inference:batch_op_rewriter_proto_cc",
        "@org_machina//machina/core/tfrt/graph_executor:config",
        "@org_machina//machina/core/tfrt/graph_executor:test_config_proto_cc",
    ],
)

cc_library(
    name = "run_options_base",
    hdrs = ["run_options_base.h"],
    deps = [
        "@com_google_absl//absl/time",
    ],
)

cc_library(
    name = "run_options",
    hdrs = ["run_options.h"],
    deps = ["//machina_serving/util:oss_or_google"] + if_oss([
        "//machina_serving/servables/machina/oss:run_options",
    ]) + if_google(["//machina_serving/servables/machina/google:run_options"]),
)

cc_library(
    name = "file_acl",
    hdrs = ["file_acl.h"],
    deps = ["//machina_serving/util:oss_or_google"] + if_oss([
        "//machina_serving/servables/machina/oss:file_acl",
    ]) + if_google(["//machina_serving/servables/machina/google:file_acl"]),
)

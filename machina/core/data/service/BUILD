load("@com_github_grpc_grpc//bazel:cc_grpc_library.bzl", "cc_grpc_library")
load(
    "//machina:machina.bzl",
    "tf_cc_test",
)
load("//machina:machina.default.bzl", "cc_header_only_library", "get_compatible_with_portable", "tf_grpc_cc_dependencies")
load(
    "//machina/core/platform:build_config.bzl",
    "tf_additional_all_protos",
    "tf_proto_library",
    "tf_protos_profiler_service",
)
load("//machina/core/platform:rules_cc.bzl", "cc_library")

package_group(
    name = "data_transfer_visibility",
    packages = [
        "//learning/brain/google/data/service/data_transfer_protocols/...",
        "//machina/core/data/google/...",
    ],
)

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = [
        "//machina:internal",
    ],
    licenses = ["notice"],
)

tf_proto_library(
    name = "common_proto",
    srcs = ["common.proto"],
    create_java_proto = False,
    create_kotlin_proto = False,
    protodeps = tf_additional_all_protos(),
)

tf_proto_library(
    name = "dispatcher_proto",
    srcs = ["dispatcher.proto"],
    has_services = 1,
    create_java_proto = False,
    create_kotlin_proto = False,
    protodeps = tf_additional_all_protos() + [
        ":common_proto",
    ],
)

tf_proto_library(
    name = "worker_proto",
    srcs = ["worker.proto"],
    has_services = 1,
    create_java_proto = False,
    create_kotlin_proto = False,
    protodeps = tf_additional_all_protos() + [
        ":common_proto",
    ],
    visibility = [
        ":data_transfer_visibility",
        "//machina:internal",
    ],
)

tf_proto_library(
    name = "export_proto",
    srcs = ["export.proto"],
    create_java_proto = False,
    create_kotlin_proto = False,
    protodeps = tf_additional_all_protos() + [
        ":common_proto",
    ],
)

cc_library(
    name = "graph_rewriters",
    srcs = ["graph_rewriters.cc"],
    hdrs = [
        "graph_rewriters.h",
    ],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        ":common",
        ":common_proto_cc",
        ":url",
        "//machina/core:protos_all_cc",
        "//machina/core/data:rewrite_utils",
        "//machina/core/framework:dataset_options_proto_cc",
        "//machina/core/framework:graph_proto_cc",
        "//machina/core/framework:node_def_proto_cc",
        "//machina/core/framework:types_proto_cc",
        "//machina/core/grappler:grappler_item",
        "//machina/core/grappler:grappler_item_builder",
        "//machina/core/grappler/clusters:virtual_cluster",
        "//machina/core/grappler/optimizers:custom_graph_optimizer",
        "//machina/core/grappler/optimizers/data:auto_shard",
        "//machina/core/grappler/optimizers/data:graph_utils",
        "//machina/core/grappler/optimizers/data:optimizer_base",
        "//machina/core/grappler/optimizers/data:remove_compression_map",
        "//machina/core/kernels/data/experimental:auto_shard_dataset_op",
        "//machina/core/platform:errors",
        "//machina/core/platform:status",
        "//machina/core/platform:statusor",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@local_xla//xla/tsl/platform:errors",
        "@local_xla//xla/tsl/platform:status",
        "@local_xla//xla/tsl/platform:statusor",
    ],
)

tf_cc_test(
    name = "graph_rewriters_test",
    size = "small",
    srcs = ["graph_rewriters_test.cc"],
    # copybara:uncomment extra_copts = ["-Wthread-safety-analysis"],
    deps = [
        ":common_proto_cc",
        ":graph_rewriters",
        ":test_util",
        "//machina/core:protos_all_cc",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core/framework:dataset_options_proto_cc",
        "//machina/core/framework:function_testlib",
        "//machina/core/framework:graph_proto_cc",
        "//machina/core/framework:node_def_proto_cc",
        "//machina/core/framework:tensor_testutil",
        "//machina/core/ops",
        "//machina/core/platform:errors",
        "//machina/core/platform:status_matchers",
        "//machina/core/platform:statusor",
        "//machina/core/platform:tstring",
        "//machina/core/platform:types",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "byte_size",
    srcs = ["byte_size.cc"],
    hdrs = ["byte_size.h"],
    deps = [
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "byte_size_test",
    srcs = ["byte_size_test.cc"],
    deps = [
        ":byte_size",
        "//machina/core:test",
        "//machina/core:test_main",
        "@com_google_absl//absl/algorithm:container",
    ],
)

cc_library(
    name = "common",
    srcs = ["common.cc"],
    hdrs = [
        "common.h",
    ],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        ":common_proto_cc",
        "//machina/core:framework",
        "//machina/core:protos_all_cc",
        "//machina/core/platform:errors",
        "//machina/core/platform:status",
        "//machina/core/platform:statusor",
        "//machina/core/platform:types",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "common_test",
    srcs = ["common_test.cc"],
    # copybara:uncomment extra_copts = ["-Wthread-safety-analysis"],
    deps = [
        ":common",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core/platform:protobuf",
        "//machina/core/platform:status_matchers",
    ],
)

cc_library(
    name = "credentials_factory",
    srcs = ["credentials_factory.cc"],
    hdrs = ["credentials_factory.h"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        "//machina/core:lib",
        "@com_google_absl//absl/strings",
    ] + tf_grpc_cc_dependencies(),
)

tf_cc_test(
    name = "credentials_factory_test",
    srcs = ["credentials_factory_test.cc"],
    # copybara:uncomment extra_copts = ["-Wthread-safety-analysis"],
    deps = [
        ":credentials_factory",
        "//machina/core:lib",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core:testlib",
    ],
)

cc_library(
    name = "cross_trainer_cache",
    hdrs = ["cross_trainer_cache.h"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        ":byte_size",
        "//machina/core:framework",
        "//machina/core/platform:errors",
        "//machina/core/platform:mutex",
        "//machina/core/platform:status",
        "//machina/core/platform:statusor",
        "//machina/core/platform:thread_annotations",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)

tf_cc_test(
    name = "cross_trainer_cache_test",
    size = "small",
    srcs = ["cross_trainer_cache_test.cc"],
    # copybara:uncomment extra_copts = ["-Wthread-safety-analysis"],
    deps = [
        ":cross_trainer_cache",
        "//machina/core:framework",
        "//machina/core:protos_all_cc",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core/framework:tensor_testutil",
        "//machina/core/lib/monitoring:cell_reader",
        "//machina/core/platform:env",
        "//machina/core/platform:errors",
        "//machina/core/platform:mutex",
        "//machina/core/platform:random",
        "//machina/core/platform:status",
        "//machina/core/platform:status_matchers",
        "//machina/core/platform:statusor",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
    ],
)

tf_cc_test(
    name = "data_service_test",
    srcs = ["data_service_test.cc"],
    # copybara:uncomment extra_copts = ["-Wthread-safety-analysis"],
    deps = [
        ":common_proto_cc",
        ":dispatcher_client",
        ":dispatcher_proto_cc",
        ":export_proto_cc",
        ":test_cluster",
        ":test_util",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core/platform:status_matchers",
        "//machina/core/platform:statusor",
        "@com_google_absl//absl/time",
    ] + tf_grpc_cc_dependencies() + tf_protos_profiler_service(),
)

cc_library(
    name = "data_transfer",
    srcs = ["data_transfer.cc"],
    hdrs = ["data_transfer.h"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    visibility = [
        ":data_transfer_visibility",
    ],
    deps = [
        ":worker_proto_cc",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core/framework:dataset_proto_cc",
        "//machina/core/framework:types_proto_cc",
        "//machina/core/platform:errors",
        "//machina/core/platform:macros",
        "//machina/core/platform:mutex",
        "//machina/core/platform:status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
    ],
)

tf_cc_test(
    name = "data_transfer_test",
    srcs = ["data_transfer_test.cc"],
    # copybara:uncomment extra_copts = ["-Wthread-safety-analysis"],
    deps = [
        ":data_transfer",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core:testlib",
        "//machina/core/framework:types_proto_cc",
        "//machina/core/platform:protobuf",
        "//machina/core/platform:status",
    ],
)

cc_library(
    name = "dataset_store",
    srcs = ["dataset_store.cc"],
    hdrs = ["dataset_store.h"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        ":common_proto_cc",
        ":dispatcher_state",
        ":utils",
        "//machina/core:lib",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "dataset_store_test",
    srcs = ["dataset_store_test.cc"],
    # copybara:uncomment extra_copts = ["-Wthread-safety-analysis"],
    deps = [
        ":common_proto_cc",
        ":dataset_store",
        "//machina/core:lib",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core:testlib",
        "//machina/core/framework:graph_proto_cc",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/memory",
    ],
)

cc_grpc_library(
    name = "dispatcher_cc_grpc_proto",
    srcs = [":dispatcher_proto"],
    compatible_with = get_compatible_with_portable(),
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    generate_mocks = True,
    grpc_only = True,
    deps = [":dispatcher_proto_cc"],
)

cc_library(
    name = "dispatcher_client",
    srcs = ["dispatcher_client.cc"],
    hdrs = ["dispatcher_client.h"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        ":common",
        ":common_proto_cc",
        ":credentials_factory",
        ":dispatcher_cc_grpc_proto",
        ":dispatcher_proto_cc",
        ":grpc_util",
        "//machina/core:framework",
        "//machina/core:framework_lite",
        "//machina/core:protos_all_cc",
        "//machina/core/framework:graph_proto_cc",
        "//machina/core/platform:errors",
        "//machina/core/platform:status",
        "//machina/core/platform:statusor",
        "@com_google_absl//absl/strings",
        "@local_xla//xla/tsl/platform:errors",
    ] + tf_grpc_cc_dependencies(),
)

tf_cc_test(
    name = "dispatcher_client_test",
    srcs = ["dispatcher_client_test.cc"],
    # copybara:uncomment extra_copts = ["-Wthread-safety-analysis"],
    deps = [
        ":common_proto_cc",
        ":data_transfer",
        ":dataset_store",
        ":dispatcher_client",
        ":test_cluster",
        ":test_util",
        "//machina/core:framework",
        "//machina/core:protos_all_cc",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core/data/service/snapshot:path_utils",
        "//machina/core/platform:errors",
        "//machina/core/platform:status_matchers",
        "//machina/core/platform:statusor",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@local_tsl//tsl/platform:path",
        "@local_xla//xla/tsl/platform:env",
        "@local_xla//xla/tsl/protobuf:protos_all_cc",
    ] + tf_grpc_cc_dependencies() + tf_protos_profiler_service(),
)

cc_library(
    name = "dispatcher_impl",
    srcs = ["dispatcher_impl.cc"],
    hdrs = [
        "dispatcher_impl.h",
    ],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        ":auto_scaler",
        ":common",
        ":common_proto_cc",
        ":credentials_factory",
        ":dataset_store",
        ":dispatcher_proto_cc",
        ":dispatcher_state",
        ":export_proto_cc",
        ":grpc_util",
        ":journal",
        ":journal_proto_cc",
        ":split_provider",
        ":task_remover",
        ":utils",
        ":validate_utils",
        ":worker_cc_grpc_proto",
        "//machina/core:core_cpu",
        "//machina/core:core_cpu_internal",
        "//machina/core:framework",
        "//machina/core:framework_internal",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core:protos_all_cc",
        "//machina/core/data:dataset_utils",
        "//machina/core/data:hash_utils",
        "//machina/core/data:snapshot_utils",
        "//machina/core/data:standalone",
        "//machina/core/data:utils",
        "//machina/core/data/service/snapshot:file_utils",
        "//machina/core/data/service/snapshot:path_utils",
        "//machina/core/data/service/snapshot:snapshot_manager",
        "//machina/core/platform:env",
        "//machina/core/platform:errors",
        "//machina/core/platform:macros",
        "//machina/core/platform:mutex",
        "//machina/core/platform:path",
        "//machina/core/platform:status",
        "//machina/core/platform:statusor",
        "//machina/core/platform:strcat",
        "//machina/core/platform:thread_annotations",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@local_xla//xla/tsl/platform:env",
        "@local_xla//xla/tsl/platform:logging",
    ] + tf_grpc_cc_dependencies(),
)

cc_library(
    name = "dispatcher_state",
    srcs = ["dispatcher_state.cc"],
    hdrs = [
        "dispatcher_state.h",
    ],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        ":common",
        ":common_proto_cc",
        ":graph_rewriters",
        ":journal",
        ":journal_proto_cc",
        "//machina/core:protos_all_cc",
        "//machina/core/platform:errors",
        "//machina/core/platform:status",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "dispatcher_state_test",
    srcs = ["dispatcher_state_test.cc"],
    # copybara:uncomment extra_copts = ["-Wthread-safety-analysis"],
    deps = [
        ":common_proto_cc",
        ":dispatcher_state",
        ":journal_proto_cc",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core:test",
        "//machina/core:test_main",
        "@com_google_absl//absl/strings",
        "@local_xla//xla/tsl/platform:status_matchers",
    ],
)

cc_library(
    name = "grpc_dispatcher_impl",
    srcs = ["grpc_dispatcher_impl.cc"],
    hdrs = ["grpc_dispatcher_impl.h"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        ":dispatcher_cc_grpc_proto",
        ":dispatcher_impl",
        ":export_proto_cc",
        "//machina/core:protos_all_cc",
        "//machina/core/distributed_runtime/rpc:grpc_util",
    ] + tf_grpc_cc_dependencies(),
)

tf_cc_test(
    name = "grpc_dispatcher_impl_test",
    srcs = ["grpc_dispatcher_impl_test.cc"],
    # copybara:uncomment extra_copts = ["-Wthread-safety-analysis"],
    deps = [
        ":common",
        ":common_proto_cc",
        ":credentials_factory",
        ":dispatcher_proto_cc",
        ":grpc_dispatcher_impl",
        ":server_lib",
        ":test_util",
        "//machina/core:protos_all_cc",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core/distributed_runtime/rpc:grpc_util",
        "//machina/core/framework:graph_proto_cc",
        "//machina/core/platform:errors",
        "//machina/core/platform:status",
        "//machina/core/platform:statusor",
        "//machina/core/platform:types",
        "@com_google_absl//absl/strings",
    ] + tf_grpc_cc_dependencies() + tf_protos_profiler_service(),
)

cc_library(
    name = "grpc_util",
    srcs = ["grpc_util.cc"],
    hdrs = [
        "grpc_util.h",
    ],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        ":common",
        "//machina/core:lib",
        "//machina/core/distributed_runtime/rpc:grpc_util",
        "@com_google_absl//absl/time",
        "@local_tsl//tsl/platform:retrying_utils",
    ] + tf_grpc_cc_dependencies(),
)

tf_cc_test(
    name = "grpc_util_test",
    srcs = ["grpc_util_test.cc"],
    # copybara:uncomment extra_copts = ["-Wthread-safety-analysis"],
    deps = [
        ":grpc_util",
        "//machina/core:lib",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core:testlib",
    ],
)

cc_library(
    name = "grpc_worker_impl",
    srcs = ["grpc_worker_impl.cc"],
    hdrs = ["grpc_worker_impl.h"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        ":export_proto_cc",
        ":worker_cc_grpc_proto",
        ":worker_impl",
        ":worker_proto_cc",
        "//machina/core:protos_all_cc",
        "//machina/core/distributed_runtime/rpc:grpc_util",
        "//machina/core/platform:errors",
        "//machina/core/platform:status",
    ] + tf_grpc_cc_dependencies(),
)

tf_cc_test(
    name = "grpc_worker_impl_test",
    srcs = ["grpc_worker_impl_test.cc"],
    # copybara:uncomment extra_copts = ["-Wthread-safety-analysis"],
    deps = [
        ":common_proto_cc",
        ":credentials_factory",
        ":dispatcher_proto_cc",
        ":grpc_worker_impl",
        ":server_lib",
        ":test_util",
        ":worker_proto_cc",
        "//machina/core:protos_all_cc",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core/distributed_runtime/rpc:grpc_util",
        "//machina/core/framework:graph_proto_cc",
        "//machina/core/framework:types_proto_cc",
        "//machina/core/platform:errors",
        "//machina/core/platform:status",
        "//machina/core/platform:statusor",
        "//machina/core/platform:types",
        "@com_google_absl//absl/strings",
    ] + tf_grpc_cc_dependencies() + tf_protos_profiler_service(),
)

cc_library(
    name = "journal",
    srcs = ["journal.cc"],
    hdrs = ["journal.h"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        ":journal_proto_cc",
        "//machina/core:lib",
        "//machina/core/platform:regexp",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)

tf_proto_library(
    name = "journal_proto",
    srcs = ["journal.proto"],
    create_java_proto = False,
    create_kotlin_proto = False,
    protodeps = [
        ":common_proto",
        "//machina/core:protos_all",
    ],
)

tf_cc_test(
    name = "journal_test",
    srcs = ["journal_test.cc"],
    # copybara:uncomment extra_copts = ["-Wthread-safety-analysis"],
    deps = [
        ":common_proto_cc",
        ":journal",
        ":journal_proto_cc",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core:testlib",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
    ],
)

cc_library(
    name = "py_utils",
    srcs = ["py_utils.cc"],
    hdrs = ["py_utils.h"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        ":credentials_factory",
    ],
)

cc_library(
    name = "server_lib",
    srcs = ["server_lib.cc"],
    hdrs = ["server_lib.h"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    linkstatic = True,
    visibility = [
        "//visibility:public",
    ],
    deps = [
        ":common_proto_cc",
        ":credentials_factory",
        ":data_transfer",
        ":export_proto_cc",
        ":grpc_dispatcher_impl",
        ":grpc_util",
        ":grpc_worker_impl",
        ":worker_client",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core/profiler/rpc:profiler_service_impl",
        "@com_google_absl//absl/strings",
    ] + tf_grpc_cc_dependencies(),
    alwayslink = 1,
)

# This needs to be cc_header_only_library - tf_pybind_cc_library_wrapper
# does not pull in the server_lib.h header.
cc_header_only_library(
    name = "server_lib_headers_lib",
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    features = ["-parse_headers"],
    deps = [
        ":dispatcher_client",
        ":grpc_util",
        ":server_lib",
    ],
)

cc_library(
    name = "split_provider",
    srcs = ["split_provider.cc"],
    hdrs = [
        "split_provider.h",
    ],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        ":common_proto_cc",
        ":dispatcher_client",
        ":grpc_util",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core/data:standalone",
        "//machina/core/platform:env",
        "//machina/core/platform:errors",
        "//machina/core/platform:mutex",
        "//machina/core/platform:status",
        "//machina/core/platform:thread_annotations",
    ],
)

tf_cc_test(
    name = "split_provider_test",
    srcs = ["split_provider_test.cc"],
    # copybara:uncomment extra_copts = ["-Wthread-safety-analysis"],
    deps = [
        ":common_proto_cc",
        ":split_provider",
        ":test_util",
        "//machina/core:framework",
        "//machina/core:test",
        "//machina/core:test_main",
        "@com_google_absl//absl/strings",
        "@local_xla//xla/tsl/lib/core:status_test_util",
        "@local_xla//xla/tsl/platform:statusor",
    ],
)

cc_library(
    name = "task_remover",
    srcs = ["task_remover.cc"],
    hdrs = ["task_remover.h"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        "//machina/core:framework_internal",
        "//machina/core:lib",
        "@com_google_absl//absl/container:flat_hash_set",
    ],
)

cc_library(
    name = "task_runner",
    srcs = ["task_runner.cc"],
    hdrs = ["task_runner.h"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        ":byte_size",
        ":common",
        ":common_proto_cc",
        ":cross_trainer_cache",
        ":data_transfer",
        ":thread_safe_buffer",
        ":worker_proto_cc",
        "//machina/core:framework",
        "//machina/core:framework_internal",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core/data:standalone",
    ],
)

tf_cc_test(
    name = "task_runner_test",
    srcs = ["task_runner_test.cc"],
    # copybara:uncomment extra_copts = ["-Wthread-safety-analysis"],
    deps = [
        ":data_transfer",
        ":task_runner",
        ":worker_proto_cc",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core:testlib",
        "//machina/core/platform:env",
        "//machina/core/platform:errors",
        "//machina/core/platform:mutex",
        "//machina/core/platform:status",
        "//machina/core/platform:status_matchers",
        "//machina/core/platform:statusor",
        "//machina/core/platform:tstring",
        "@com_google_absl//absl/memory",
    ],
)

cc_library(
    name = "test_cluster",
    testonly = True,
    srcs = ["test_cluster.cc"],
    hdrs = ["test_cluster.h"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        ":common_proto_cc",
        ":data_transfer",
        ":dispatcher_client",
        ":dispatcher_proto_cc",
        ":export_proto_cc",
        ":server_lib",
        ":test_util",
        ":worker_client",
        ":worker_proto_cc",
        "//machina/core:framework",
        "//machina/core:protos_all_cc",
        "//machina/core:tflite_portable_logging",
        "//machina/core/platform:errors",
        "//machina/core/platform:status",
        "//machina/core/platform:statusor",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@local_xla//xla/tsl/platform:env",
    ],
)

cc_library(
    name = "test_data_transfer",
    testonly = True,
    srcs = ["test_data_transfer.cc"],
    deps = [
        ":data_transfer",
        "@com_google_absl//absl/status",
    ],
    alwayslink = 1,
)

cc_library(
    name = "test_util",
    testonly = True,
    srcs = ["test_util.cc"],
    hdrs = [
        "test_util.h",
    ],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    data = ["//machina/core/data/service/testdata"],
    deps = [
        ":common_proto_cc",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core:test",
        "//machina/core/data:dataset_test_base",
        "//machina/core/framework:function_proto_cc",
        "//machina/core/framework:function_testlib",
        "//machina/core/framework:graph_proto_cc",
        "//machina/core/framework:node_def_proto_cc",
        "//machina/core/framework:tensor_testutil",
        "//machina/core/framework:types_proto_cc",
        "//machina/core/platform:path",
        "//machina/core/platform:tstring",
        "//machina/core/platform:types",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:protobuf",
        "@local_xla//xla/tsl/platform:errors",
    ],
)

tf_cc_test(
    name = "test_util_test",
    srcs = ["test_util_test.cc"],
    # copybara:uncomment extra_copts = ["-Wthread-safety-analysis"],
    deps = [
        ":common_proto_cc",
        ":test_util",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core/data:dataset_test_base",
        "//machina/core/data:standalone",
        "//machina/core/framework:graph_proto_cc",
        "//machina/core/framework:tensor_testutil",
        "//machina/core/platform:status_matchers",
        "//machina/core/platform:types",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "thread_safe_buffer",
    hdrs = ["thread_safe_buffer.h"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        "//machina/core:framework_lite",
        "//machina/core/platform:macros",
        "//machina/core/platform:status",
        "//machina/core/platform:statusor",
    ],
)

tf_cc_test(
    name = "thread_safe_buffer_test",
    size = "small",
    srcs = ["thread_safe_buffer_test.cc"],
    # copybara:uncomment extra_copts = ["-Wthread-safety-analysis"],
    shard_count = 3,
    deps = [
        ":thread_safe_buffer",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core/framework:tensor_testutil",
        "//machina/core/platform:status_matchers",
        "//machina/core/platform:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "url",
    srcs = ["url.cc"],
    hdrs = ["url.h"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        "//machina/core/platform:regexp",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "url_test",
    size = "small",
    srcs = ["url_test.cc"],
    # copybara:uncomment extra_copts = ["-Wthread-safety-analysis"],
    deps = [
        ":url",
        "//machina/core:test",
        "//machina/core:test_main",
    ],
)

cc_library(
    name = "utils",
    srcs = ["utils.cc"],
    hdrs = ["utils.h"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        ":common_proto_cc",
        "//machina/core:lib",
        "@com_google_absl//absl/memory",
    ],
)

tf_cc_test(
    name = "utils_test",
    srcs = ["utils_test.cc"],
    # copybara:uncomment extra_copts = ["-Wthread-safety-analysis"],
    deps = [
        ":common_proto_cc",
        ":utils",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core:testlib",
    ],
)

cc_library(
    name = "validate_utils",
    srcs = ["validate_utils.cc"],
    hdrs = ["validate_utils.h"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        "//machina/core:protos_all_cc",
        "//machina/core/platform:errors",
        "//machina/core/platform:protobuf",
        "//machina/core/platform:status",
        "//machina/core/platform:statusor",
    ],
)

tf_cc_test(
    name = "validate_utils_test",
    size = "small",
    srcs = ["validate_utils_test.cc"],
    # copybara:uncomment extra_copts = ["-Wthread-safety-analysis"],
    deps = [
        ":validate_utils",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core/framework:tensor_shape_proto_cc",
        "//machina/core/platform:status_matchers",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/strings",
    ],
)

cc_grpc_library(
    name = "worker_cc_grpc_proto",
    srcs = [":worker_proto"],
    compatible_with = get_compatible_with_portable(),
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    generate_mocks = True,
    grpc_only = True,
    deps = [":worker_proto_cc"],
)

cc_library(
    name = "worker_client",
    srcs = ["worker_client.cc"],
    hdrs = ["worker_client.h"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        ":common",
        ":common_proto_cc",
        ":credentials_factory",
        ":data_transfer",
        ":grpc_util",
        ":worker_cc_grpc_proto",
        ":worker_impl",
        ":worker_proto_cc",
        "//machina/core:framework",
        "//machina/core:framework_lite",
        "//machina/core/framework:dataset_proto_cc",
        "//machina/core/framework:types_proto_cc",
        "//machina/core/platform:errors",
        "//machina/core/platform:status",
        "//machina/core/platform:statusor",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:platform_port",
        "@local_xla//xla/tsl/platform:errors",
    ] + tf_grpc_cc_dependencies(),
)

tf_cc_test(
    name = "worker_client_test",
    size = "small",
    srcs = ["worker_client_test.cc"],
    # copybara:uncomment extra_copts = ["-Wthread-safety-analysis"],
    deps = [
        ":common",
        ":common_proto_cc",
        ":data_transfer",
        ":dispatcher_client",
        ":dispatcher_proto_cc",
        ":server_lib",
        ":test_cluster",
        ":test_util",
        ":worker_client",
        ":worker_impl",
        ":worker_proto_cc",
        "//machina/core:framework",
        "//machina/core:protos_all_cc",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core/framework:graph_proto_cc",
        "//machina/core/framework:tensor_testutil",
        "//machina/core/platform:errors",
        "//machina/core/platform:status",
        "//machina/core/platform:status_matchers",
        "//machina/core/platform:statusor",
        "//machina/core/platform:types",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
    ] + tf_grpc_cc_dependencies() + tf_protos_profiler_service(),
)

cc_library(
    name = "worker_impl",
    srcs = ["worker_impl.cc"],
    hdrs = [
        "worker_impl.h",
    ],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        ":byte_size",
        ":common",
        ":common_proto_cc",
        ":data_transfer",
        ":dispatcher_client",
        ":dispatcher_proto_cc",
        ":export_proto_cc",
        ":graph_rewriters",
        ":grpc_util",
        ":split_provider",
        ":task_runner",
        ":utils",
        ":worker_proto_cc",
        "//machina/core:core_cpu",
        "//machina/core:core_cpu_internal",
        "//machina/core:framework",
        "//machina/core:framework_internal",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core:protos_all_cc",
        "//machina/core/data:standalone",
        "//machina/core/data/service/snapshot:path_utils",
        "//machina/core/data/service/snapshot:snapshot_split_provider",
        "//machina/core/data/service/snapshot:snapshot_stream_writer",
        "//machina/core/platform:env",
        "//machina/core/platform:errors",
        "//machina/core/platform:logging",
        "//machina/core/platform:platform_port",
        "//machina/core/platform:status",
        "//machina/core/platform:statusor",
        "//machina/core/platform:thread_annotations",
        "//machina/core/platform:types",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@local_xla//xla/tsl/platform:status_to_from_proto",
        "@local_xla//xla/tsl/protobuf:protos_all_cc",
    ] + tf_grpc_cc_dependencies(),
)

tf_cc_test(
    name = "worker_impl_test",
    srcs = ["worker_impl_test.cc"],
    # copybara:uncomment extra_copts = ["-Wthread-safety-analysis"],
    deps = [
        ":test_cluster",
        ":worker_impl",
        "//machina/core:test",
        "//machina/core:test_main",
    ] + tf_protos_profiler_service(),
)

cc_library(
    name = "auto_scaler",
    srcs = ["auto_scaler.cc"],
    hdrs = ["auto_scaler.h"],
    deps = [
        "//machina/core:framework",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@local_tsl//tsl/platform:mutex",
        "@local_tsl//tsl/platform:thread_annotations",
        "@local_xla//xla/tsl/platform:status",
    ],
)

tf_cc_test(
    name = "auto_scaler_test",
    srcs = ["auto_scaler_test.cc"],
    deps = [
        ":auto_scaler",
        "//machina/core:framework",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core/lib/monitoring:cell_reader",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest_main",
        "@local_xla//xla/tsl/platform:status_matchers",
    ],
)

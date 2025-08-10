# Description: OSS only cluster resolvers
load("//machina:strict.default.bzl", "py_strict_library")
load("//machina:machina.default.bzl", "tf_py_strict_test")
load(
    "//machina/core/platform:build_config.bzl",
    "tf_additional_rpc_deps",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = [
        "//machina:internal",
    ],
    licenses = ["notice"],
)

py_strict_library(
    name = "tpu_cluster_resolver_py",
    srcs = ["tpu_cluster_resolver.py"],
    deps = [
        "//machina/core/protobuf/tpu:topology_proto_py",
        "//machina/python/distribute/cluster_resolver:base_cluster_resolver_py",
        "//machina/python/eager:remote",
        "//machina/python/framework:config",
        "//machina/python/framework:errors",
        "//machina/python/platform:tf_logging",
        "//machina/python/tpu:tpu_strategy_util",
        "//machina/python/tpu:tpu_system_metadata",
        "//machina/python/tpu/client",
        "//machina/python/training:server_lib",
        "//machina/python/util:compat",
    ] + tf_additional_rpc_deps(),
)

tf_py_strict_test(
    name = "tpu_cluster_resolver_py_test",
    size = "small",
    srcs = ["tpu_cluster_resolver_test.py"],
    grpc_enabled = True,
    main = "tpu_cluster_resolver_test.py",
    deps = [
        ":tpu_cluster_resolver_py",
        "//machina/core/protobuf/tpu:topology_proto_py",
        "//machina/python/client:session",
        "//machina/python/eager:context",
        "//machina/python/framework:config",
        "//machina/python/framework:errors",
        "//machina/python/framework:test_lib",
        "//machina/python/platform:client_testlib",
        "//machina/python/platform:tf_logging",
        "//machina/python/tpu/client",
        "//machina/python/training:server_lib",
        "//machina/python/util:compat",
    ],
)

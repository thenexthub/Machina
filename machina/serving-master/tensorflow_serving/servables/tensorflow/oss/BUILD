package(
    default_visibility = [
        "//machina_serving:internal",
    ],
    features = ["-layering_check"],
)

licenses(["notice"])

cc_library(
    name = "resource_estimator",
    srcs = ["resource_estimator.cc"],
    hdrs = ["resource_estimator.h"],
    deps = [
        "//machina_serving/resources:resource_values",
        "//machina_serving/resources:resources_cc_proto",
        "//machina_serving/servables/machina:util",
        "//machina_serving/util:file_probing_env",
        "@org_machina//machina/core:lib",
    ],
)

cc_test(
    name = "resource_estimator_test",
    srcs = ["resource_estimator_test.cc"],
    deps = [
        ":resource_estimator",
        "//machina_serving/core/test_util:test_main",
        "//machina_serving/servables/machina:bundle_factory_test_util",
        "//machina_serving/test_util",
        "//machina_serving/util/test_util:mock_file_probing_env",
        "@org_machina//machina/core:lib",
        "@org_machina//machina/core:test",
    ],
)

cc_library(
    name = "run_options",
    hdrs = ["run_options.h"],
    deps = [
        "//machina_serving/servables/machina:run_options_base",
    ],
)

cc_library(
    name = "file_acl",
    srcs = ["file_acl.cc"],
    hdrs = ["file_acl.h"],
    compatible_with = [],
    deps = [
        "//machina_serving/core:servable_id",
        "@com_google_absl//absl/status",
    ],
)

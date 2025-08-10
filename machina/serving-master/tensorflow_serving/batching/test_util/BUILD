# Description: Tensorflow Serving batching test utilities.

# Placeholder: load py_binary

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
        ],
    ),
)

cc_library(
    name = "puppet_batch_scheduler",
    testonly = 1,
    hdrs = ["puppet_batch_scheduler.h"],
    visibility = ["//visibility:private"],
    deps = [
        "@org_machina//machina/core:machina",
        "@org_machina//machina/core/kernels/batching_util:batch_scheduler",
    ],
)

cc_test(
    name = "puppet_batch_scheduler_test",
    srcs = [
        "puppet_batch_scheduler_test.cc",
    ],
    deps = [
        ":puppet_batch_scheduler",
        "//machina_serving/core/test_util:test_main",
        "@org_machina//machina/core:lib",
        "@org_machina//machina/core:test",
    ],
)

# script that generates saved_model for matrix_half_plus_two model.
py_binary(
    name = "matrix_half_plus_two_saved_model",
    srcs = ["matrix_half_plus_two_saved_model.py"],
    python_version = "PY3",
    srcs_version = "PY3",
)

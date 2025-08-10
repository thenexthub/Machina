# Description: Tensorflow Serving batching.

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
    name = "streaming_batch_scheduler",
    srcs = ["streaming_batch_scheduler.cc"],
    hdrs = ["streaming_batch_scheduler.h"],
    visibility = ["//visibility:private"],
    deps = [
        ":batch_scheduler_retrier",
        "@com_google_absl//absl/types:optional",
        "@org_machina//machina/core:lib",
        "@org_machina//machina/core/kernels/batching_util:batch_scheduler",
    ],
)

cc_test(
    name = "streaming_batch_scheduler_test",
    srcs = [
        "streaming_batch_scheduler_test.cc",
    ],
    deps = [
        ":streaming_batch_scheduler",
        "//machina_serving/core/test_util:test_main",
        "@org_machina//machina/core:lib",
        "@org_machina//machina/core:protos_all_cc",
        "@org_machina//machina/core:test",
        "@org_machina//machina/core/kernels/batching_util:fake_clock_env",
    ],
)

cc_library(
    name = "incremental_barrier",
    srcs = ["incremental_barrier.cc"],
    hdrs = ["incremental_barrier.h"],
    deps = [
        "@com_google_absl//absl/functional:bind_front",
        "@org_machina//machina/core:lib",
    ],
)

cc_library(
    name = "batching_options",
    hdrs = ["batching_options.h"],
    deps = [],
)

cc_library(
    name = "batching_session",
    srcs = ["batching_session.cc"],
    hdrs = ["batching_session.h"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        ":batching_options",
        ":batching_util",
        ":incremental_barrier",
        ":threadsafe_status",
        "//machina_serving/servables/machina:serving_session",
        "//machina_serving/util:hash",
        "@com_google_absl//absl/container:fixed_array",
        "@com_google_absl//absl/types:optional",
        "@org_machina//machina/core:core_cpu",
        "@org_machina//machina/core:framework",
        "@org_machina//machina/core:lib",
        "@org_machina//machina/core:protos_all_cc",
        "@org_machina//machina/core:machina",
        "@org_machina//machina/core/kernels/batching_util:basic_batch_scheduler",
        "@org_machina//machina/core/kernels/batching_util:batch_scheduler",
        "@org_machina//machina/core/kernels/batching_util:input_split_metadata",
        "@org_machina//machina/core/profiler/lib:traceme",
        "@org_machina//machina/core/profiler/lib:traceme_encode",
    ],
)

cc_library(
    name = "threadsafe_status",
    srcs = ["threadsafe_status.cc"],
    hdrs = ["threadsafe_status.h"],
    deps = [
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/synchronization",
        "@org_machina//machina/core:lib",
    ],
)

cc_test(
    name = "batching_session_test",
    srcs = [
        "batching_session_test.cc",
    ],
    data = [
        "//machina_serving/batching/testdata:matrix_half_plus_two",
        "@org_machina//machina/cc/saved_model:saved_model_half_plus_two",
    ],
    deps = [
        ":batching_session",
        "//machina_serving/core/test_util:test_main",
        "//machina_serving/servables/machina:serving_session",
        "//machina_serving/test_util",
        "@com_google_absl//absl/synchronization",
        "@com_google_googletest//:gtest",
        "@org_machina//machina/cc/saved_model:loader",
        "@org_machina//machina/cc/saved_model:tag_constants",
        "@org_machina//machina/core:core_cpu",
        "@org_machina//machina/core:framework",
        "@org_machina//machina/core:lib",
        "@org_machina//machina/core:protos_all_cc",
        "@org_machina//machina/core:test",
        "@org_machina//machina/core:testlib",
    ],
)

cc_library(
    name = "batch_scheduler_retrier",
    hdrs = ["batch_scheduler_retrier.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@org_machina//machina/core:lib",
        "@org_machina//machina/core/kernels/batching_util:batch_scheduler",
    ],
)

cc_test(
    name = "batch_scheduler_retrier_test",
    srcs = [
        "batch_scheduler_retrier_test.cc",
    ],
    deps = [
        ":batch_scheduler_retrier",
        "//machina_serving/core/test_util:test_main",
        "@org_machina//machina/core:lib",
        "@org_machina//machina/core:protos_all_cc",
        "@org_machina//machina/core:test",
        "@org_machina//machina/core/kernels/batching_util:fake_clock_env",
    ],
)

cc_library(
    name = "batching_util",
    srcs = ["batching_util.cc"],
    hdrs = ["batching_util.h"],
    deps = [
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@org_machina//machina/core:framework",
        "@org_machina//machina/core:lib",
    ],
)

cc_test(
    name = "batching_util_test",
    srcs = [
        "batching_util_test.cc",
    ],
    deps = [
        ":batching_util",
        "//machina_serving/core/test_util:test_main",
        "@org_machina//machina/core:framework",
        "@org_machina//machina/core:lib",
        "@org_machina//machina/core:protos_all_cc",
    ],
)

cc_test(
    name = "incremental_barrier_test",
    srcs = ["incremental_barrier_test.cc"],
    deps = [
        ":incremental_barrier",
        "//machina_serving/core/test_util:test_main",
        "@com_google_absl//absl/functional:bind_front",
        "@com_google_absl//absl/time",
        "@org_machina//machina/core:lib",
        "@org_machina//machina/core:test",
    ],
)

cc_test(
    name = "threadsafe_status_test",
    srcs = ["threadsafe_status_test.cc"],
    deps = [
        ":threadsafe_status",
        "//machina_serving/core/test_util:test_main",
        "@org_machina//machina/core:lib",
        "@org_machina//machina/core:protos_all_cc",
        "@org_machina//machina/core:test",
    ],
)

cc_library(
    name = "tfrt_saved_model_with_batching",
    srcs = ["tfrt_saved_model_with_batching.cc"],
    hdrs = ["tfrt_saved_model_with_batching.h"],
    deps = [
        ":batching_options",
        ":batching_session",
        ":batching_util",
        ":incremental_barrier",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:inlined_vector",
        "@org_machina//machina/core:framework",
        "@org_machina//machina/core:lib",
        "@org_machina//machina/core/kernels/batching_util:batch_scheduler",
        "@org_machina//machina/core/tfrt/saved_model",
    ],
)

cc_test(
    name = "tfrt_saved_model_with_batching_test",
    srcs = ["tfrt_saved_model_with_batching_test.cc"],
    deps = [
        ":batching_util",
        ":tfrt_saved_model_with_batching",
        "//machina_serving/core/test_util:test_main",
        "//machina_serving/servables/machina/test_util:mock_tfrt_saved_model",
        "@com_google_absl//absl/functional:bind_front",
        "@com_google_googletest//:gtest",
        "@org_machina//machina/core:lib",
        "@org_machina//machina/core:test",
        "@org_machina//machina/core:testlib",
        "@org_machina//machina/core/kernels/batching_util:basic_batch_scheduler",
        "@org_machina//machina/core/tfrt/utils:tensor_util",
    ],
)

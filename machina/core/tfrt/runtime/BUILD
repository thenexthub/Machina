load("//machina:machina.bzl", "tf_cc_shared_test", "tf_cc_test")
load("//machina/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = [":friends"],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    packages = [
        # Authorized users go here.
        "//machina/compiler/mlir/tfrt/...",
        "//machina/core/tfrt/...",
        "//machina/core/runtime_fallback/...",
        # copybara:uncomment "//machina_serving/...",
        # copybara:uncomment "//cloud/ai/platform/prediction/...",
        # copybara:uncomment "//learning/brain/experimental/tfrt/...",
        # copybara:uncomment "//learning/brain/tfrt/...",
        # copybara:uncomment "//learning/infra/mira/...",
        # copybara:uncomment "//learning/pathways/serving/...",
        # copybara:uncomment "//learning/serving/...",
        # copybara:uncomment "//smartass/brain/...",
        # copybara:uncomment "//quality/webanswers/servo2/...",
    ],
)

cc_library(
    name = "runtime",
    srcs = ["runtime.cc"],
    hdrs = ["runtime.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":work_queue_interface",
        "//machina/core:core_cpu_base",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core/protobuf:for_core_protos_cc",
        "//machina/core/runtime_fallback/kernel:kernel_fallback_tensor",
        "//machina/core/tfrt/graph_executor:graph_execution_options",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:string_view",
        "@tf_runtime//:core_runtime_alwayslink",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:tensor",
        "@tf_runtime//backends/cpu:core_runtime",
        "@tf_runtime//backends/cpu:core_runtime_alwayslink",
    ],
)

tf_cc_shared_test(
    name = "runtime_test",
    srcs = ["runtime_test.cc"],
    tags = ["no_oss"],
    deps = [
        ":runtime",
        "//machina/c:tf_tensor",
        "//machina/core:test",
        "@com_google_googletest//:gtest_main",
        "@tf_runtime//cpp_tests:common",
    ],
)

cc_library(
    name = "work_queue_interface",
    srcs = ["work_queue_interface.cc"],
    hdrs = ["work_queue_interface.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//machina/core:lib",
        "//machina/core/platform:threadpool_interface",
        "//machina/core/profiler/lib:connected_traceme",
        "//machina/core/profiler/lib:traceme_encode",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
    ],
)

tf_cc_test(
    name = "work_queue_interface_test",
    srcs = ["work_queue_interface_test.cc"],
    deps = [
        ":work_queue_interface",
        "//machina/core:test",
        "//machina/core/tfrt/utils:thread_pool",
        "@com_google_googletest//:gtest_main",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//cpp_tests:common",
    ],
)

cc_library(
    name = "tf_threadpool_concurrent_work_queue",
    srcs = ["tf_threadpool_concurrent_work_queue.cc"],
    hdrs = ["tf_threadpool_concurrent_work_queue.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":work_queue_interface",
        "//machina/core:framework_lite",
        "//machina/core:lib",
        "//machina/core/platform:errors",
        "//machina/core/platform:status",
        "//machina/core/platform:threadpool_interface",
        "//machina/core/tfrt/utils:thread_pool",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/status:statusor",
        "@toolchain-project//toolchain:Support",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
    ],
)

cc_library(
    name = "stream",
    srcs = ["stream.cc"],
    hdrs = ["stream.h"],
    deps = [
        ":step_id",
        "//machina/compiler/mlir/machina",
        "//machina/core/framework:tensor",
        "//machina/core/framework:tensor_proto_cc",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/utility",
        "@toolchain-project//mlir:IR",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:random",
        "@local_tsl//tsl/platform:threadpool_interface",
        "@local_tsl//tsl/profiler/lib:traceme",
    ],
)

cc_library(
    name = "step_id",
    srcs = ["step_id.cc"],
    hdrs = ["step_id.h"],
    deps = [
        "//machina/core/tfrt/kernels:stream_ops_util_constants",
        "@com_google_absl//absl/strings:str_format",
    ],
)

tf_cc_shared_test(
    name = "stream_test",
    srcs = ["stream_test.cc"],
    tags = ["no_oss"],
    deps = [
        ":step_id",
        ":stream",
        "//machina/core/framework:tensor",
        "//machina/core/framework:tensor_testutil",
        "//machina/core/tfrt/saved_model:saved_model_testutil",
        "//machina/core/tfrt/utils:thread_pool",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:statusor",
    ],
)

tf_cc_test(
    name = "tf_threadpool_concurrent_work_queue_test",
    srcs = ["tf_threadpool_concurrent_work_queue_test.cc"],
    deps = [
        ":tf_threadpool_concurrent_work_queue",
        "//machina/core:framework_internal",
        "//machina/core:lib",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core/platform:errors",
        "//machina/core/platform:status_matchers",
        "//machina/core/tfrt/utils:thread_pool",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
    ],
)

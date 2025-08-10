load("//machina:machina.bzl", "if_google", "tf_cc_test")
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
        # copybara:uncomment "//learning/infra/mira/...",
        # copybara:uncomment "//learning/serving/...",
        # copybara:uncomment "//learning/brain/experimental/tfrt/...",
        # copybara:uncomment "//learning/brain/tfrt/cpp_tests/tpu_model/...",
        # copybara:uncomment "//learning/brain/tfrt/...",
        "//machina/core/tfrt/...",
        "//machina/core/runtime_fallback/runtime/...",
        "//machina_serving/...",
    ],
)

cc_library(
    name = "run_handler_util",
    srcs = ["run_handler_util.cc"],
    hdrs = ["run_handler_util.h"],
    deps = [
        "//machina/core:lib",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "run_handler_util_test",
    size = "small",
    srcs = ["run_handler_util_test.cc"],
    deps = [
        ":run_handler_util",
        "//machina/core:framework_internal",
        "//machina/core:lib",
        "//machina/core:test",
        "//machina/core:test_main",
    ],
)

cc_library(
    name = "run_handler",
    srcs = ["run_handler.cc"],
    hdrs = ["run_handler.h"],
    deps = [
        ":run_handler_util",
        "//machina/core:framework_internal",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core/profiler/lib:connected_traceme",
        "//machina/core/profiler/lib:traceme",
        "//machina/core/profiler/lib:traceme_encode",
        "//machina/core/protobuf:for_core_protos_cc",
        "//machina/core/tfrt/runtime:work_queue_interface",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@eigen_archive//:eigen3",
        "@local_xla//xla/tsl/platform:env",
        "@tf_runtime//:hostcontext",
    ],
)

tf_cc_test(
    name = "run_handler_test",
    size = "small",
    srcs = ["run_handler_test.cc"],
    deps = [
        ":run_handler",
        "//machina/core:framework_internal",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core/common_runtime:direct_session_internal",
        "//machina/core/kernels:cwise_op",
        "//machina/core/kernels:matmul_op",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_googletest//:gtest",
        "@eigen_archive//:eigen3",
        "@tf_runtime//:hostcontext",
    ],
)

cc_library(
    name = "run_handler_concurrent_work_queue",
    srcs = [
        "run_handler_concurrent_work_queue.cc",
    ] + if_google(
        ["//third_party/tf_runtime_eigen:concurrent_work_queue_hdrs"],
        ["@tf_runtime//third_party/concurrent_work_queue:concurrent_work_queue_hdrs"],
    ),
    hdrs = ["run_handler_concurrent_work_queue.h"],
    deps = [
        ":run_handler",
        "//machina/core/platform:strcat",
        "//machina/core/tfrt/runtime:work_queue_interface",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@toolchain-project//toolchain:Support",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
    ],
)

tf_cc_test(
    name = "run_handler_concurrent_work_queue_test",
    srcs = ["run_handler_concurrent_work_queue_test.cc"],
    deps = [
        ":run_handler_concurrent_work_queue",
        "//machina/core:framework_internal",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core:test_main",
        "//machina/core/platform:errors",
        "//machina/core/platform:status_matchers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@local_xla//xla/tsl/protobuf:error_codes_proto_impl_cc",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
    ],
)

load(
    "//machina:machina.bzl",
    "tf_cc_test",
    "tf_cuda_cc_test",
    "tf_cuda_library",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = [
        "//machina:internal",
    ],
    licenses = ["notice"],
)

tf_cuda_library(
    name = "coordination_test_opkernel_registration",
    testonly = 1,
    srcs = ["coordination_test_opkernel_registration.cc"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        "//machina/core:framework",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:cord",
        "@local_xla//xla/tsl/distributed_runtime/coordination:coordination_service_agent",
        "@local_xla//xla/tsl/distributed_runtime/coordination:coordination_service_error_util",
    ],
    alwayslink = 1,
)

tf_cuda_cc_test(
    name = "c_api_coordination_test",
    size = "medium",
    srcs = ["c_api_coordination_test.cc"],
    # copybara:uncomment extra_copts = ["-Wthread-safety-analysis"],
    tags = [
        "no_cuda_asan",  # TODO(b/193450885)
        "no_oss",  # TODO(b/283120771)
        "no_windows",  # TODO(b/207281588)
    ],
    deps = [
        ":coordination_test_opkernel_registration",
        "//machina/c:c_api_experimental",
        "//machina/c/eager:c_api",
        "//machina/c/eager:c_api_experimental",
        "//machina/c/eager:c_api_internal",
        "//machina/c/eager:c_api_test_util",
        "//machina/c/eager:tfe_tensorhandle_internal",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core/distributed_runtime:server_lib",
        "//machina/core/platform:env",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@local_xla//xla/tsl/lib/core:status_test_util",
    ],
)

tf_cuda_cc_test(
    name = "c_api_session_coordination_test",
    size = "medium",
    srcs = ["c_api_session_coordination_test.cc"],
    # copybara:uncomment extra_copts = ["-Wthread-safety-analysis"],
    tags = [
        "no_windows",  # TODO(b/207281588)
    ],
    deps = [
        ":coordination_test_opkernel_registration",
        "//machina/c:c_api",
        "//machina/c:c_api_experimental",
        "//machina/c:c_test_util",
        "//machina/c/eager:c_api_test_util",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core/distributed_runtime:server_lib",
        "//machina/core/distributed_runtime/rpc:grpc_session",
        "//machina/core/platform:env",
    ],
)

tf_cuda_cc_test(
    name = "c_api_multi_client_test",
    size = "medium",
    srcs = ["c_api_multi_client_test.cc"],
    # copybara:uncomment extra_copts = ["-Wthread-safety-analysis"],
    tags = [
        "no_windows",  # TODO(b/207281588)
    ],
    deps = [
        "//machina/c:c_api_experimental",
        "//machina/c:c_test_util",
        "//machina/c/eager:c_api",
        "//machina/c/eager:c_api_experimental",
        "//machina/c/eager:c_api_internal",
        "//machina/c/eager:c_api_test_util",
        "//machina/c/eager:tfe_context_internal",
        "//machina/c/eager:tfe_tensorhandle_internal",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core/common_runtime/eager:context",
        "//machina/core/distributed_runtime:server_lib",
        "//machina/core/platform:env",
        "@local_xla//xla/tsl/distributed_runtime/coordination:coordination_service",
        "@local_xla//xla/tsl/distributed_runtime/coordination:coordination_service_agent",
    ],
)

tf_cc_test(
    name = "c_api_multi_client_function_test",
    size = "small",
    srcs = ["c_api_multi_client_function_test.cc"],
    # copybara:uncomment extra_copts = ["-Wthread-safety-analysis"],
    tags = [
        "no_oss",  # test uses TFRT
    ],
    deps = [
        "//machina/c:c_api_experimental",
        "//machina/c:c_test_util",
        "//machina/c/eager:c_api",
        "//machina/c/eager:c_api_experimental",
        "//machina/c/eager:c_api_internal",
        "//machina/c/eager:c_api_test_util",
        "//machina/c/eager:tfe_context_internal",
        "//machina/c/eager:tfe_op_internal",
        "//machina/c/eager:tfe_tensorhandle_internal",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core/common_runtime/eager:context",
        "//machina/core/common_runtime/eager:eager_operation",
        "//machina/core/common_runtime/eager:kernel_and_device",
        "//machina/core/distributed_runtime:server_lib",
        "//machina/core/platform:env",
        "@com_google_absl//absl/synchronization",
        "@local_xla//xla/tsl/distributed_runtime/coordination:coordination_service",
        "@local_xla//xla/tsl/distributed_runtime/coordination:coordination_service_agent",
    ],
)

tf_cuda_cc_test(
    name = "c_api_recoverable_jobs_test",
    size = "medium",
    srcs = ["c_api_recoverable_jobs_test.cc"],
    # copybara:uncomment extra_copts = ["-Wthread-safety-analysis"],
    tags = [
        "no_cuda_asan",  # TODO(b/193450885)
        "no_windows",  # TODO(b/207281588)
    ],
    deps = [
        ":coordination_test_opkernel_registration",
        "//machina/c:c_api_experimental",
        "//machina/c/eager:c_api",
        "//machina/c/eager:c_api_experimental",
        "//machina/c/eager:c_api_internal",
        "//machina/c/eager:c_api_test_util",
        "//machina/c/eager:tfe_tensorhandle_internal",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core/distributed_runtime:server_lib",
        "//machina/core/platform:env",
        "@com_google_absl//absl/time",
        "@local_xla//xla/tsl/lib/core:status_test_util",
    ],
)

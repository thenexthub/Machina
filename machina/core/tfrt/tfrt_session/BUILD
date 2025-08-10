load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//machina:strict.default.bzl", "py_strict_library")
load("//machina:machina.bzl", "if_google", "tf_cc_shared_test")
load("//machina:machina.default.bzl", "tf_py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    licenses = ["notice"],
)

cc_library(
    name = "tfrt_session_init",
    hdrs = ["tfrt_session_init.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//machina/core:lib",
        "//machina/core/common_runtime:local_session_selection",
    ],
)

cc_library(
    name = "tfrt_session",
    srcs = ["tfrt_session.cc"],
    hdrs = ["tfrt_session.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//machina/compiler/mlir/tfrt:backend_compiler",
        "//machina/compiler/mlir/tfrt:tfrt_compile_options",
        "//machina/core:core_cpu",
        "//machina/core:framework",
        "//machina/core:framework_lite",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core/common_runtime:local_session_selection",
        "//machina/core/common_runtime:process_util",
        "//machina/core/common_runtime:session_factory",
        "//machina/core/platform:errors",
        "//machina/core/platform:status",
        "//machina/core/platform:statusor",
        "//machina/core/platform:threadpool_interface",
        "//machina/core/platform:threadpool_options",
        "//machina/core/runtime_fallback/runtime:runtime_fallback_alwayslink",
        "//machina/core/tfrt/fallback:fallback_state",
        "//machina/core/tfrt/graph_executor",
        "//machina/core/tfrt/graph_executor:graph_execution_options",
        "//machina/core/tfrt/mlrt/interpreter:context",
        "//machina/core/tfrt/mlrt/kernel",
        "//machina/core/tfrt/mlrt/kernel:batch_kernel",
        "//machina/core/tfrt/run_handler_thread_pool:run_handler_concurrent_work_queue",
        "//machina/core/tfrt/runtime",
        "//machina/core/tfrt/runtime:tf_threadpool_concurrent_work_queue",
        "//machina/core/tfrt/runtime:work_queue_interface",
        "//machina/core/tfrt/utils",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/log:die_if_null",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@eigen_archive//:eigen3",
        "@toolchain-project//toolchain:Support",
        "@local_xla//xla/tsl/platform:errors",
        "@tf_runtime//:core_runtime",
        "@tf_runtime//:hostcontext",
    ] + if_google([
        "//learning/brain/tfrt/tpu:tfrttpu_alwayslink",
        "//learning/brain/tfrt/tpu/fused_kernel:tpu_fused_kernel_alwayslink",
        "//learning/brain/tfrt/support:export_mlir",
    ]),
    alwayslink = 1,
)

tf_cc_shared_test(
    name = "tfrt_session_test",
    size = "small",
    srcs = ["tfrt_session_test.cc"],
    data = [
        "//machina/core/tfrt/saved_model/tests:toy_v1/1/saved_model.pb",
        "//machina/core/tfrt/saved_model/tests:toy_v1/1/variables/variables.data-00000-of-00001",
        "//machina/core/tfrt/saved_model/tests:toy_v1/1/variables/variables.index",
    ],
    tags = ["no_oss"],
    deps = [
        ":tfrt_session",
        "//machina/cc:array_ops",
        "//machina/cc:cc_ops",
        "//machina/cc:const_op",
        "//machina/cc:ops",
        "//machina/cc:scope",
        "//machina/cc/saved_model:reader",
        "//machina/core:core_cpu",
        "//machina/core:core_cpu_base",
        "//machina/core:framework",
        "//machina/core:framework_lite",
        "//machina/core:protos_all_cc",
        "//machina/core:test",
        "//machina/core/common_runtime:local_session_selection",
        "//machina/core/common_runtime:process_util",
        "//machina/core/common_runtime:session_factory",
        "//machina/core/framework:tensor_testutil",
        "//machina/core/platform:resource_loader",
        "//machina/core/platform:status",
        "//machina/core/platform:threadpool_options",
        "//machina/core/tfrt/runtime",
        "//machina/core/tfrt/saved_model:saved_model_testutil",
        "//machina/core/tfrt/utils:thread_pool",
        "//machina/python/framework:test_ops_kernels",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest",
        "@local_tsl//tsl/platform:protobuf",
    ],
)

# copybara:uncomment_begin(C++ Global registration not working in python, b/290988709)
# py_strict_library(
#     name = "tfrt_session_py",
#     visibility = ["//visibility:public"],
#     deps = [
#         "//machina/core/tfrt/tfrt_session",
#     ],
# )
#
# tf_py_strict_test(
#     name = "tfrt_session_python_test",
#     srcs = ["tfrt_session_python_test.py"],
#     exec_properties = select({
#         "//tools/cpp:asan_build": {"cpp_link.mem": "16g"},
#         "//conditions:default": None,
#     }),
#     deps = [
#         ":tfrt_session_py",
#         "//machina/core:protos_all_py",
#         "//machina/python/client:session",
#         "//machina/python/framework:constant_op",
#         "//machina/python/framework:ops",
#         "//machina/python/framework:test_lib",
#         "//machina/python/ops:math_ops",
#         "//machina/python/platform:test",
#     ],
# )
# copybara:uncomment_end

load("//machina:strict.default.bzl", "py_strict_binary", "py_strict_library", "py_strict_test")
load("//machina:machina.default.bzl", "cuda_py_strict_test", "tf_py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

py_strict_library(
    name = "profiler",
    srcs = ["profiler.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":model_analyzer",
        ":option_builder",
        ":tfprof_logger",
        "//machina/core/profiler:protos_all_py",
        "//machina/python/util:tf_export",
    ],
)

py_strict_library(
    name = "profiler_client",
    srcs = ["profiler_client.py"],
    deps = [
        "//machina/python/framework:errors",
        "//machina/python/profiler/internal:_pywrap_profiler_plugin",
        "//machina/python/util:tf_export",
    ],
)

cuda_py_strict_test(
    name = "profiler_client_test",
    srcs = ["profiler_client_test.py"],
    tags = ["no_pip"],
    deps = [
        ":profiler_client",
        ":profiler_v2",
        "//machina/python/eager:test",
        "//machina/python/framework:errors",
        "//machina/python/framework:test_lib",
        "@pypi_portpicker//:pkg",
    ],
)

py_strict_library(
    name = "profiler_v2",
    srcs = ["profiler_v2.py"],
    visibility = ["//machina:internal"],
    deps = [
        "//machina/python/framework:errors",
        "//machina/python/platform:tf_logging",
        "//machina/python/profiler/internal:_pywrap_profiler",
        "//machina/python/util:tf_export",
    ],
)

cuda_py_strict_test(
    name = "profiler_v2_test",
    srcs = ["profiler_v2_test.py"],
    tags = [
        "no_pip",
    ],
    deps = [
        ":profiler_v2",
        ":trace",
        "//machina/python/eager:test",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:errors",
        "//machina/python/framework:test_lib",
        "//machina/python/platform:gfile",
    ],
)

py_strict_test(
    name = "profiler_wrapper_test",
    srcs = ["profiler_wrapper_test.py"],
    tags = [
        "no_pip",
    ],
    deps = [
        "//machina/python/eager:test",
        "//machina/python/framework:test_lib",
        "//machina/python/profiler/internal:_pywrap_profiler_plugin",
    ],
)

py_strict_library(
    name = "option_builder",
    srcs = ["option_builder.py"],
    deps = [
        ":tfprof_logger",
        "//machina/python/util:tf_export",
    ],
)

py_strict_library(
    name = "model_analyzer",
    srcs = ["model_analyzer.py"],
    deps = [
        ":option_builder",
        ":tfprof_logger",
        "//machina/core/profiler:protos_all_py",
        "//machina/python/eager:context",
        "//machina/python/framework:errors",
        "//machina/python/framework:ops",
        "//machina/python/util:_pywrap_tfprof",
        "//machina/python/util:tf_export",
    ],
)

cuda_py_strict_test(
    name = "model_analyzer_test",
    srcs = ["model_analyzer_test.py"],
    tags = [
        "no_pip",
        "notap",
        "oss_serial",
    ],
    xla_enable_strict_auto_jit = False,  # Node names are different with autojit
    deps = [
        ":model_analyzer",
        ":option_builder",
        ":profile_context",
        "//machina/core:protos_all_py",
        "//machina/core/profiler:protos_all_py",
        "//machina/python:distributed_framework_test_lib",
        "//machina/python/client:session",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:gradients",
        "//machina/python/ops:random_ops",
        "//machina/python/ops:variables",
        "//machina/python/ops:while_loop",
        "//machina/python/platform:client_testlib",
        "//machina/python/platform:gfile",
        "//machina/python/profiler/internal:model_analyzer_testlib",
        "//machina/python/util:compat",
    ],
)

cuda_py_strict_test(
    name = "profiler_test",
    srcs = ["profiler_test.py"],
    tags = ["no_pip"],
    xla_enable_strict_auto_jit = False,  # Node names are different with autojit
    deps = [
        ":model_analyzer",
        ":option_builder",
        "//machina/core:protos_all_py",
        "//machina/python/client:session",
        "//machina/python/framework:ops",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:variables",
        "//machina/python/platform:client_testlib",
        "//machina/python/platform:gfile",
        "//machina/python/profiler/internal:model_analyzer_testlib",
    ],
)

py_strict_library(
    name = "tfprof_logger",
    srcs = ["tfprof_logger.py"],
    deps = [
        "//machina/core/profiler:protos_all_py",
        "//machina/python/eager:context",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor_shape",
        "//machina/python/platform:gfile",
        "//machina/python/profiler/internal:flops_registry",
        "//machina/python/util:tf_export",
    ],
)

tf_py_strict_test(
    name = "tfprof_logger_test",
    size = "small",
    srcs = ["tfprof_logger_test.py"],
    deps = [
        "//machina/python/framework:constant_op",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/platform:client_testlib",
    ],
)

py_strict_library(
    name = "profile_context",
    srcs = ["profile_context.py"],
    deps = [
        ":model_analyzer",
        "//machina/core:protos_all_py",
        "//machina/python/client:session",
        "//machina/python/framework:errors",
        "//machina/python/framework:ops",
        "//machina/python/platform:gfile",
        "//machina/python/util:_pywrap_tfprof",
        "//machina/python/util:compat",
    ],
)

cuda_py_strict_test(
    name = "profile_context_test",
    srcs = ["profile_context_test.py"],
    tags = [
        "no_gpu",  # b/136036359
        "no_pip",
    ],
    xla_enable_strict_auto_jit = False,  # Node names are different with autojit
    deps = [
        ":option_builder",
        ":profile_context",
        "//machina/python/client:session",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:variables",
        "//machina/python/platform:client_testlib",
        "//machina/python/platform:gfile",
        "//machina/python/profiler/internal:model_analyzer_testlib",
    ],
)

py_strict_library(
    name = "pprof_profiler",
    srcs = ["pprof_profiler.py"],
    deps = ["@com_google_pprof//:pprof_proto_py"],
)

py_strict_test(
    name = "pprof_profiler_test",
    size = "small",
    srcs = ["pprof_profiler_test.py"],
    main = "pprof_profiler_test.py",
    tags = ["no_pip"],  # TODO(annarev): get it working with pip.
    deps = [
        ":pprof_profiler",
        "//machina/core:protos_all_py",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:while_loop",
        "//machina/python/platform:client_testlib",
        "@com_google_pprof//:pprof_proto_py",
    ],
)

py_strict_library(
    name = "trace",
    srcs = ["trace.py"],
    visibility = [
        "//perftools/accelerators/xprof/xprofilez/integration_tests:__pkg__",
        "//machina:internal",
    ],
    deps = [
        "//machina/python/profiler/internal:_pywrap_traceme",
        "//machina/python/util:tf_export",
    ],
)

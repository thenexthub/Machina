load("@bazel_skylib//:bzl_library.bzl", "bzl_library")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("//machina:machina.default.bzl", "get_compatible_with_portable")
load("//machina/lite:build_def.bzl", "tflite_copts", "tflite_copts_warnings", "tflite_self_contained_libs_test_suite")
load("//machina/lite:special_rules.bzl", "internal_visibility_allowlist", "tflite_portable_test_suite")
load("//machina/lite/core:special_rules.bzl", "core_cc_api_stable_visibility_allowlist", "macros_visibility_allowlist")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    licenses = ["notice"],
)

exports_files(
    srcs = [
        "create_op_resolver.h",
        "macros.h",
        "subgraph.h",
    ],
    visibility = [
        "//machina/lite:__subpackages__",
    ],
)

bzl_library(
    name = "special_rules_bzl",
    srcs = ["special_rules.bzl"],
    visibility = ["//machina/lite:__subpackages__"],
)

# The public target for the C++ API excluding experimental APIs.
# TODO(ahentz): investigate dependency on gemm_support requiring usage of tf_copts.
cc_library(
    name = "framework_stable",
    srcs = [
        "subgraph.h",
    ],
    hdrs = [
        "interpreter.h",
        "interpreter_builder.h",
        "macros.h",
        "model.h",
        "model_builder.h",
        "signature_runner.h",
    ],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts() + tflite_copts_warnings(),
    visibility = ["//machina/lite:__subpackages__"],
    deps = [
        ":cc_api_stable",
        ":signature_runner",
        "//machina/compiler/mlir/lite/core:model_builder_base",
        "//machina/compiler/mlir/lite/experimental/remat:metadata_util",
        "//machina/lite:allocation",
        "//machina/lite:array",
        "//machina/lite:external_cpu_backend_context",
        "//machina/lite:graph_info",
        "//machina/lite:interpreter_options_header",
        "//machina/lite:macros",
        "//machina/lite:memory_planner",
        "//machina/lite:mutable_op_resolver",
        "//machina/lite:stderr_reporter",
        "//machina/lite:string",
        "//machina/lite:type_to_tflitetype",
        "//machina/lite:util",
        "//machina/lite/c:common_internal",
        "//machina/lite/core/api",
        "//machina/lite/core/async:async_signature_runner",
        "//machina/lite/core/c:common",
        "//machina/lite/experimental/resource",
        "//machina/lite/internal:signature_def",
        "//machina/lite/profiling:root_profiler",
        "//machina/lite/profiling/telemetry:profiler",
        "//machina/lite/profiling/telemetry/c:telemetry_setting",
        "//machina/lite/profiling/telemetry/c:telemetry_setting_internal",
        "//machina/lite/schema:schema_fbs",
        "@flatbuffers//:runtime_cc",
    ],
)

# The public target for the full C++ API, including experimental APIs.
#
# Experimental APIs are functional, tested and usable in production; however,
# the corresponding API surface has not been finalized, and is subject to
# change.
alias(
    name = "framework",
    actual = "framework_experimental",
    compatible_with = get_compatible_with_portable(),
    visibility = ["//visibility:public"],
)

# The full C++ API, including experimental APIs.
#
# Experimental APIs are functional, tested and usable in production; however,
# the corresponding API surface has not been finalized, and is subject to
# change.
#
# Note that if you have code which depends on both stable and experimental API
# features, it's fine to depend only on 'framework_experimental', since
# that includes 'framework_stable' as a subset.
cc_library(
    name = "framework_experimental",
    srcs = [],
    hdrs = [
        "interpreter.h",
        "interpreter_builder.h",
        "macros.h",
        "model.h",
        "model_builder.h",
        "subgraph.h",
    ],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts() + tflite_copts_warnings(),
    visibility = [
        "//machina/lite:__subpackages__",
    ],
    deps = [
        ":cc_api_experimental",
        ":cc_api_stable",
        ":model_builder",
        ":signature_runner",
        "//machina/compiler/mlir/lite/core:model_builder_base",
        "//machina/compiler/mlir/lite/experimental/remat:metadata_util",
        "//machina/lite:allocation",
        "//machina/lite:array",
        "//machina/lite:external_cpu_backend_context",
        "//machina/lite:graph_info",
        "//machina/lite:interpreter_options_header",
        "//machina/lite:macros",
        "//machina/lite:memory_planner",
        "//machina/lite:mutable_op_resolver",
        "//machina/lite:stderr_reporter",
        "//machina/lite:string",
        "//machina/lite:type_to_tflitetype",
        "//machina/lite:util",
        "//machina/lite/c:common_internal",
        "//machina/lite/core/api",
        "//machina/lite/core/async:async_signature_runner",
        "//machina/lite/core/c:common",
        "//machina/lite/experimental/resource",
        "//machina/lite/internal:signature_def",
        "//machina/lite/profiling:root_profiler",
        "//machina/lite/profiling/telemetry:profiler",
        "//machina/lite/profiling/telemetry/c:telemetry_setting",
        "//machina/lite/profiling/telemetry/c:telemetry_setting_internal",
        "//machina/lite/schema:schema_fbs",
        "@flatbuffers//:runtime_cc",
    ],
    alwayslink = 1,  # TODO(b/161243354): eliminate this.
)

# This is a private target, its visibility is set to public only to be
# used by LiteRT dependencies.
# Do not use this target directly and don't consider it as a part of the public API.
# TODO(weiyiw): Refactor LiteRT deps from TFLite.
alias(
    name = "private_cc_api_stable",
    actual = ":cc_api_stable",
    tags = ["avoid_dep"],
    visibility = [
        "//visibility:public",
    ],
)

# TODO(b/242310498): move logger.cc from machina/lite/ to here.
cc_library(
    name = "cc_api_stable",
    srcs = [
        "interpreter.cc",
        "interpreter_builder.cc",
        "subgraph.h",
    ],
    hdrs = [
        "interpreter.h",
        "interpreter_builder.h",
        "model.h",
        "model_builder.h",
        "signature_runner.h",
    ],
    compatible_with = get_compatible_with_portable(),
    visibility = [
        "//machina/lite:__subpackages__",
        "//third_party/odml/litert/litert:__subpackages__",
    ] + core_cc_api_stable_visibility_allowlist(),
    deps = [
        ":model_builder",
        ":signature_runner",
        ":subgraph",
        "//machina/compiler/mlir/lite/core:model_builder_base",
        "//machina/compiler/mlir/lite/experimental/remat:metadata_util",
        "//machina/compiler/mlir/lite/schema:schema_fbs",
        "//machina/compiler/mlir/lite/schema:schema_utils",
        "//machina/lite:allocation",
        "//machina/lite:array",
        "//machina/lite:external_cpu_backend_context",
        "//machina/lite:graph_info",
        "//machina/lite:interpreter_options_header",
        "//machina/lite:macros",
        "//machina/lite:memory_planner",
        "//machina/lite:minimal_logging",
        "//machina/lite:mutable_op_resolver",
        "//machina/lite:shared_library",
        "//machina/lite:simple_memory_arena",
        "//machina/lite:stderr_reporter",
        "//machina/lite:string",
        "//machina/lite:machina_profiler_logger_shim",
        "//machina/lite:type_to_tflitetype",
        "//machina/lite:util",
        "//machina/lite:version",
        "//machina/lite/c:common_internal",
        "//machina/lite/core/api",
        "//machina/lite/core/api:verifier",
        "//machina/lite/core/async:async_signature_runner",
        "//machina/lite/core/c:c_api_types",
        "//machina/lite/core/c:common",
        "//machina/lite/delegates:telemetry",
        "//machina/lite/delegates/xnnpack:tflite_with_xnnpack_qs8",
        "//machina/lite/delegates/xnnpack:tflite_with_xnnpack_qu8",
        "//machina/lite/experimental/resource",
        "//machina/lite/internal:signature_def",
        "//machina/lite/kernels/internal:compatibility",
        "//machina/lite/profiling:platform_profiler",
        "//machina/lite/profiling:root_profiler",
        "//machina/lite/profiling/telemetry",
        "//machina/lite/profiling/telemetry:profiler",
        "//machina/lite/profiling/telemetry/c:telemetry_setting",
        "//machina/lite/profiling/telemetry/c:telemetry_setting_internal",
        "//machina/lite/schema:conversion_metadata_fbs",
        "//machina/lite/schema:schema_fbs",
        "//machina/lite/schema:schema_utils",
        "@flatbuffers//:runtime_cc",
        "@ruy//ruy:denormal",
    ],
    alwayslink = 1,
)

# The key parts of the C++ API.  This target defines the TF Lite classes for
# loading models and interpreting them.
# DEPRECATED: prefer to depend on :cc_api_stable or :cc_api_experimental.
alias(
    name = "cc_api",
    actual = "cc_api_experimental",
    visibility = ["//machina/lite:__subpackages__"],
)

# The key parts of the C++ API, including experimental APIs.
#
# This target has restricted visibility; for a public target that exposes
# these APIs, see 'framework_experimental' above.
cc_library(
    name = "cc_api_experimental",
    srcs = [
        "interpreter_experimental.cc",
    ],
    hdrs = [
        "interpreter.h",
        "interpreter_builder.h",
        "model.h",
        "model_builder.h",
        "signature_runner.h",
        "subgraph.h",
    ],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts() + tflite_copts_warnings(),
    visibility = [
        "//machina/lite:__subpackages__",
    ],
    deps = [
        ":cc_api_stable",
        ":signature_runner",
        "//machina/compiler/mlir/lite/core:model_builder_base",
        "//machina/compiler/mlir/lite/experimental/remat:metadata_util",
        "//machina/lite:allocation",
        "//machina/lite:array",
        "//machina/lite:external_cpu_backend_context",
        "//machina/lite:graph_info",
        "//machina/lite:interpreter_options_header",
        "//machina/lite:macros",
        "//machina/lite:memory_planner",
        "//machina/lite:mutable_op_resolver",
        "//machina/lite:stderr_reporter",
        "//machina/lite:string",
        "//machina/lite:type_to_tflitetype",
        "//machina/lite:util",
        "//machina/lite/c:common_internal",
        "//machina/lite/core/api",
        "//machina/lite/core/async:async_signature_runner",
        "//machina/lite/core/c:c_api_types",
        "//machina/lite/core/c:common",
        "//machina/lite/experimental/resource",
        "//machina/lite/internal:signature_def",
        "//machina/lite/profiling:root_profiler",
        "//machina/lite/profiling/telemetry:profiler",
        "//machina/lite/profiling/telemetry/c:telemetry_setting",
        "//machina/lite/profiling/telemetry/c:telemetry_setting_internal",
        "//machina/lite/schema:schema_fbs",
        "@flatbuffers//:runtime_cc",
    ],
    alwayslink = 1,  # TODO(b/161243354): eliminate this.
)

cc_library(
    name = "model_builder",
    hdrs = ["model_builder.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts_warnings(),
    visibility = internal_visibility_allowlist(),
    deps = [
        "//machina/compiler/mlir/lite/core:model_builder_base",
        "//machina/lite:stderr_reporter",
        "//machina/lite/core/api:error_reporter",
    ],
    alwayslink = 1,
)

cc_library(
    name = "signature_runner",
    srcs = ["signature_runner.cc"],
    hdrs = ["signature_runner.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = [
        "//machina/lite:__pkg__",
        "//machina/lite/core:__subpackages__",
        "//third_party/odml/infra/genai/inference/executor/google_tensor:__subpackages__",
    ],
    deps = [
        "//machina/lite/c:common",
        "//machina/lite/core:subgraph",
        "//machina/lite/core/c:c_api_types",
        "//machina/lite/core/c:common",
        "//machina/lite/internal:signature_def",
    ],
)

# Test signature runner.
cc_test(
    name = "signature_runner_test",
    size = "small",
    srcs = ["signature_runner_test.cc"],
    data = [
        "//machina/lite:testdata/multi_signatures.bin",
        "//machina/lite:testdata/reverse_signature_model.bin",
    ],
    deps = [
        ":framework",
        ":signature_runner",
        "//machina/compiler/mlir/lite/core:model_builder_base",
        "//machina/lite:model_builder",
        "//machina/lite/core/kernels:builtin_ops",
        "//machina/lite/testing:util",
        "@com_google_googletest//:gtest_main",
    ],
)

# Test model framework.
cc_test(
    name = "model_test",
    size = "small",
    srcs = ["model_test.cc"],
    data = [
        "//machina/lite:testdata/0_subgraphs.bin",
        "//machina/lite:testdata/2_subgraphs.bin",
        "//machina/lite:testdata/2_subgraphs_dont_delegate_name.bin",
        "//machina/lite:testdata/add_shared_tensors.bin",
        "//machina/lite:testdata/empty_model.bin",
        "//machina/lite:testdata/multi_add_flex.bin",
        "//machina/lite:testdata/segment_sum_invalid_buffer.bin",
        "//machina/lite:testdata/sparse_tensor.bin",
        "//machina/lite:testdata/test_min_runtime.bin",
        "//machina/lite:testdata/test_model.bin",
        "//machina/lite:testdata/test_model_broken.bin",
        "//machina/lite:testdata/test_model_redux_precision.bin",
        "//machina/lite:testdata/while_op_with_forwarding_input.bin",
        "//machina/lite:testdata/zero_size_constant.bin",
    ],
    tags = [
        "no_windows",  # TODO(b/194459105): the test is flaky.
        "noasan",
        "tflite_not_portable",
        "tflite_smoke_test",
    ],
    deps = [
        ":framework",
        "//machina/compiler/mlir/lite:allocation",
        "//machina/lite:framework",
        "//machina/lite:string_util",
        "//machina/lite/core/api",
        "//machina/lite/core/api:verifier",
        "//machina/lite/core/c:common",
        "//machina/lite/core/kernels:builtin_ops",
        "//machina/lite/schema:schema_fbs",
        "//machina/lite/testing:util",
        "@com_google_googletest//:gtest_main",
        "@flatbuffers//:runtime_cc",
    ],
)

cc_library(
    name = "create_op_resolver_header",
    hdrs = [
        "create_op_resolver.h",
    ],
    compatible_with = get_compatible_with_portable(),
    visibility = [
        "//machina/lite:__subpackages__",
    ],
    deps = [
        "//machina/lite:mutable_op_resolver",
        "//machina/lite:op_resolver",
    ],
)

# Defines CreateOpResolver with all builtin ops.
cc_library(
    name = "create_op_resolver_with_builtin_ops",
    srcs = ["create_op_resolver_with_builtin_ops.cc"],
    hdrs = ["create_op_resolver.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts(),
    visibility = [
        "//machina/lite:__subpackages__",
    ],
    deps = [
        "//machina/lite:mutable_op_resolver",
        "//machina/lite:op_resolver",
        "//machina/lite/core/kernels:builtin_ops",
    ],
    # Some targets only have an implicit dependency on CreateOpResolver.
    # This avoids warnings about backwards references when linking.
    alwayslink = True,
)

# This target is only for use by the "tflite_custom_c_library" and "tflite_custom_cc_library" build
# macro and should not be used anywhere other than in the implementation of that build macro.
# "tflite_custom_c_library" requires target to be public, that's why we duplicated
# :create_op_resolver_header target to be used only by "tflite_custom_c_library".
# Making :create_op_resolver_header public could cause some problems because it is widely used
# inside the TF Lite code base, that might lead others outside the TF Lite code base to copy that
# dependency and use it and subsequently depend on it, which would be bad. Using a separate
# :private_create_op_resolver_header target ensures that the only use of the unwantedly-"public"
# target is inside the "tflite_custom_c_library" itself, where it is less likely to get copied into
# third party code.
alias(
    name = "private_create_op_resolver_header",
    actual = ":create_op_resolver_header",
    tags = ["avoid_dep"],
    visibility = [
        "//visibility:public",
    ],
)

cc_library(
    name = "macros",
    hdrs = ["macros.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = [
        "//machina/lite:__subpackages__",
    ] + macros_visibility_allowlist(),
)

cc_library(
    name = "subgraph",
    srcs = [
        "subgraph.cc",
    ],
    hdrs = [
        "subgraph.h",
    ],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts() + tflite_copts_warnings(),
    visibility = [
        "//platforms/darwinn/tflite:__subpackages__",
        "//machina/lite:__subpackages__",
        "//machina/lite/core:__subpackages__",
        "//machina/lite/kernels:__subpackages__",
    ],
    deps = [
        "//machina/compiler/mlir/lite/experimental/remat:metadata_util",
        "//machina/lite:allocation",
        "//machina/lite:array",
        "//machina/lite:graph_info",
        "//machina/lite:interpreter_options_header",
        "//machina/lite:kernel_api",
        "//machina/lite:macros",
        "//machina/lite:memory_planner",
        "//machina/lite:minimal_logging",
        "//machina/lite:util",
        "//machina/lite/c:common_internal",
        "//machina/lite/core/api",
        "//machina/lite/core/c:c_api_types",
        "//machina/lite/core/c:common",
        "//machina/lite/experimental/resource",
        "//machina/lite/profiling:root_profiler",
        "//machina/lite/profiling/telemetry",
        "//machina/lite/schema:schema_fbs",
    ] + select({
        "//machina/lite:tflite_use_simple_memory_planner": [
            "//machina/lite:simple_planner",
        ],
        "//conditions:default": [
            "//machina/lite:arena_planner",
        ],
    }) + select({
        "//machina/lite:machina_profiler_config": [
            "//machina/lite:machina_profiler_logger_shim",
        ],
        "//conditions:default": [],
    }),
    alwayslink = 1,  # TODO(b/161243354): eliminate this.
)

# Test subgraph.
cc_test(
    name = "subgraph_test",
    size = "small",
    srcs = [
        "subgraph_test.cc",
    ],
    deps = [
        ":framework_stable",
        "//machina/lite:framework",
        "//machina/lite:util",
        "//machina/lite/c:c_api_types",
        "//machina/lite/kernels:builtin_ops",  # build_cleaner: keep
        "@com_google_absl//absl/log:check",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "model_building",
    srcs = ["model_building.cc"],
    hdrs = ["model_building.h"],
    visibility = [
        "//machina/lite:__subpackages__",
    ],
    deps = [
        ":subgraph",
        "//machina/lite:array",
        "//machina/lite:framework",
        "//machina/lite:type_to_tflitetype",
        "//machina/lite/c:c_api_types",
        "//machina/lite/c:common",
        "//machina/lite/core/c:common",
        "//machina/lite/kernels:builtin_ops",
        "//machina/lite/schema:schema_fbs",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)

cc_test(
    name = "model_building_test",
    srcs = ["model_building_test.cc"],
    deps = [
        ":model_building",
        "//machina/lite:framework",
        "//machina/lite/c:c_api_types",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "subgraph_composite_inlining_test",
    srcs = ["subgraph_composite_inlining_test.cc"],
    deps = [
        ":model_building",
        ":subgraph",
        "//machina/lite:array",
        "//machina/lite:builtin_ops",
        "//machina/lite:framework",
        "//machina/lite:util",
        "//machina/lite/c:c_api_types",
        "//machina/lite/c:common",
        "//machina/lite/core/c:common",
        "//machina/lite/kernels:kernel_util",
        "//machina/lite/kernels:subgraph_test_util",
        "//machina/lite/kernels:test_util",
        "//machina/lite/kernels/internal:tensor_ctypes",
        "//machina/lite/schema:schema_fbs",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

tflite_self_contained_libs_test_suite(name = "self_contained_libs_test_suite")

tflite_portable_test_suite()

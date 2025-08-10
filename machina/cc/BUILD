# Description:
# TensorFlow is a computational framework, primarily for use in machine
# learning applications.

load(
    "//machina:machina.bzl",
    "cc_library_with_android_deps",
    "tf_cc_test",
    "tf_copts",
    "transitive_hdrs",
)
load("//machina:machina.default.bzl", "filegroup", "tf_gen_op_wrappers_cc")
load("//machina/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

filegroup(
    name = "srcs_no_runtime",
    srcs = [
        "framework/gradients.h",
        "framework/ops.h",
        "framework/scope.h",
        "framework/scope_internal.h",
        "//machina/cc/saved_model:loader.h",
    ],
)

filegroup(
    name = "srcs",
    srcs = [
        "framework/gradients.h",
        "framework/ops.h",
        "framework/scope.h",
        "framework/scope_internal.h",
        "ops/array_ops.h",
        "ops/while_loop.h",
        "//machina/cc/saved_model:loader.h",
    ],
)

filegroup(
    name = "pywrap_required_hdrs",
    srcs = [
        "training/coordinator.h",
    ],
    visibility = ["//machina/python/grappler:__pkg__"],
)

cc_library(
    name = "gradients",
    srcs = [
        "framework/gradients.cc",
        "framework/while_gradients.cc",
        "framework/while_gradients.h",
    ],
    hdrs = ["framework/gradients.h"],
    deps = [
        ":cc_ops",
        ":cc_ops_internal",
        ":grad_op_registry",
        ":ops",
        ":scope",
        ":scope_internal",
        ":while_loop",
        "//machina/core:core_cpu",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:lib_internal",
    ],
)

tf_cc_test(
    name = "framework_gradients_test",
    srcs = ["framework/gradients_test.cc"],
    deps = [
        ":cc_ops",
        ":client_session",
        ":grad_op_registry",
        ":grad_ops",
        ":gradients",
        ":testutil",
        "//machina/core:all_kernels",
        "//machina/core:framework",
        "//machina/core:framework_internal",
        "//machina/core:protos_all_cc",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core:testlib",
    ],
)

tf_cc_test(
    name = "framework_while_gradients_test",
    size = "small",
    srcs = ["framework/while_gradients_test.cc"],
    deps = [
        ":cc_ops",
        ":client_session",
        ":grad_op_registry",
        ":grad_ops",
        ":gradients",
        ":testutil",
        ":while_loop",
        "//machina/core:all_kernels",
        "//machina/core:framework",
        "//machina/core:framework_internal",
        "//machina/core:protos_all_cc",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core:testlib",
    ],
)

cc_library(
    name = "gradient_checker",
    srcs = ["framework/gradient_checker.cc"],
    hdrs = ["framework/gradient_checker.h"],
    deps = [
        ":cc_ops",
        ":client_session",
        ":gradients",
        ":ops",
        ":scope",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:lib_internal",
    ],
)

tf_cc_test(
    name = "framework_gradient_checker_test",
    srcs = ["framework/gradient_checker_test.cc"],
    deps = [
        ":cc_ops",
        ":grad_op_registry",
        ":grad_ops",
        ":gradient_checker",
        ":testutil",
        "//machina/core:all_kernels",
        "//machina/core:framework",
        "//machina/core:framework_internal",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core:testlib",
    ],
)

cc_library(
    name = "grad_ops",
    deps = [
        ":array_grad",
        ":data_flow_grad",
        ":functional_grad",
        ":image_grad",
        ":linalg_grad",
        ":manip_grad",
        ":math_grad",
        ":nn_grad",
        ":resource_variable_grad",
    ],
)

cc_library(
    name = "grad_testutil",
    testonly = 1,
    srcs = ["gradients/grad_testutil.cc"],
    hdrs = ["gradients/grad_testutil.h"],
    deps = [
        ":grad_op_registry",
        ":ops",
        ":scope",
        "@com_google_absl//absl/status",
    ],
)

cc_library_with_android_deps(
    name = "ops",
    srcs = ["framework/ops.cc"],
    hdrs = ["framework/ops.h"],
    android_deps = ["//machina/core:portable_machina_lib"],
    deps = [
        "//machina/core:core_cpu",
        "//machina/core:framework",
        "//machina/core:graph",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core:ops",
        "//machina/core:protos_all_cc",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)

cc_library_with_android_deps(
    name = "scope",
    srcs = [
        "framework/scope.cc",
        "framework/scope_internal.h",
    ],
    hdrs = ["framework/scope.h"],
    android_deps = ["//machina/core:portable_machina_lib"],
    common_deps = [
        ":ops",
    ],
    deps = [
        "//machina/core:core_cpu",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "@com_google_absl//absl/strings",
    ],
)

cc_library_with_android_deps(
    name = "scope_internal",
    hdrs = ["framework/scope_internal.h"],
    common_deps = [
        ":scope",
    ],
    deps = [],
)

tf_cc_test(
    name = "framework_scope_test",
    srcs = ["framework/scope_test.cc"],
    deps = [
        ":ops",
        ":scope",
        "//machina/cc:cc_ops",
        "//machina/core:framework",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core:testlib",
    ],
)

cc_library_with_android_deps(
    name = "client_session",
    srcs = ["client/client_session.cc"],
    hdrs = ["client/client_session.h"],
    android_deps = ["//machina/core:portable_machina_lib"],
    common_deps = [
        ":ops",
        ":scope",
    ],
    deps = [
        "//machina/core:core_cpu",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
    ],
)

tf_cc_test(
    name = "client_client_session_test",
    srcs = ["client/client_session_test.cc"],
    deps = [
        ":cc_ops",
        ":client_session",
        "//machina/core:all_kernels",
        "//machina/core:core_cpu_internal",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:machina",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core:testlib",
        "@com_google_absl//absl/synchronization",
        "@eigen_archive//:eigen3",
    ],
)

cc_library_with_android_deps(
    name = "const_op",
    srcs = ["ops/const_op.cc"],
    hdrs = ["ops/const_op.h"],
    android_deps = [
        "//machina/core:portable_machina_lib",
    ],
    common_deps = [
        ":ops",
        ":scope",
    ],
    deps = [
        "//machina/core:core_cpu",
        "//machina/core:framework",
    ],
)

tf_cc_test(
    name = "ops_const_op_test",
    srcs = ["ops/const_op_test.cc"],
    deps = [
        ":const_op",
        "//machina/core:framework",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core:testlib",
    ],
)

cc_library_with_android_deps(
    name = "while_loop",
    srcs = ["ops/while_loop.cc"],
    hdrs = ["ops/while_loop.h"],
    android_deps = [
        "//machina/core:portable_machina_lib",
    ],
    common_deps = [
        ":cc_ops",
        ":cc_ops_internal",
        ":ops",
        ":scope",
        ":scope_internal",
    ],
    deps = [
        "//machina/core:core_cpu",
        "//machina/core:framework",
    ],
)

tf_cc_test(
    name = "ops_while_loop_test",
    size = "small",
    srcs = ["ops/while_loop_test.cc"],
    deps = [
        ":cc_ops",
        ":client_session",
        ":testutil",
        ":while_loop",
        "//machina/core:core_cpu",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core:testlib",
    ],
)

cc_library(
    name = "grad_op_registry",
    srcs = ["framework/grad_op_registry.cc"],
    hdrs = ["framework/grad_op_registry.h"],
    deps = [
        ":ops",
        ":scope",
    ],
)

cc_library(
    name = "array_grad",
    srcs = ["gradients/array_grad.cc"],
    deps = [
        ":cc_ops",
        ":cc_ops_internal",
        ":grad_op_registry",
        ":gradients",
        "//machina/core:lib_proto_parsing",
        "//machina/core:protos_all_cc",
        "@com_google_absl//absl/status",
    ],
    alwayslink = 1,
)

tf_cc_test(
    name = "gradients_array_grad_test",
    srcs = ["gradients/array_grad_test.cc"],
    deps = [
        ":array_grad",
        ":cc_ops",
        ":cc_ops_internal",
        ":grad_op_registry",
        ":grad_testutil",
        ":gradient_checker",
        ":testutil",
        "//machina/core:protos_all_cc",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core:testlib",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "image_grad",
    srcs = ["gradients/image_grad.cc"],
    deps = [
        ":cc_ops",
        ":cc_ops_internal",
        ":grad_op_registry",
        ":gradients",
        "//machina/core:protos_all_cc",
        "@com_google_absl//absl/status",
    ],
    alwayslink = 1,
)

tf_cc_test(
    name = "gradients_image_grad_test",
    srcs = ["gradients/image_grad_test.cc"],
    deps = [
        ":cc_ops",
        ":client_session",
        ":grad_op_registry",
        ":grad_testutil",
        ":gradient_checker",
        ":image_grad",
        ":testutil",
        "//machina/core:lib_internal",
        "//machina/core:protos_all_cc",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core:testlib",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "grad_helper",
    srcs = ["gradients/grad_helper.cc"],
    hdrs = ["gradients/grad_helper.h"],
    visibility = ["//machina:internal"],
    deps = [
        ":cc_ops",
        "//machina/cc:ops",
        "//machina/cc:scope",
    ],
)

cc_library(
    name = "linalg_grad",
    srcs = ["gradients/linalg_grad.cc"],
    deps = [
        ":cc_ops",
        ":cc_ops_internal",
        ":grad_helper",
        ":grad_op_registry",
        ":gradients",
        "//machina/core:protos_all_cc",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
    ],
    alwayslink = 1,
)

tf_cc_test(
    name = "gradients_linalg_grad_test",
    srcs = ["gradients/linalg_grad_test.cc"],
    deps = [
        ":cc_ops",
        ":client_session",
        ":grad_op_registry",
        ":grad_testutil",
        ":gradient_checker",
        ":gradients",
        ":linalg_grad",
        ":testutil",
        "//machina/core:lib_internal",
        "//machina/core:protos_all_cc",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core:testlib",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "math_grad",
    srcs = ["gradients/math_grad.cc"],
    deps = [
        ":cc_ops",
        ":cc_ops_internal",
        ":grad_helper",
        ":grad_op_registry",
        ":gradients",
        "//machina/core:protos_all_cc",
        "@com_google_absl//absl/status",
    ],
    alwayslink = 1,
)

tf_cc_test(
    name = "gradients_math_grad_test",
    srcs = ["gradients/math_grad_test.cc"],
    deps = [
        ":cc_ops",
        ":client_session",
        ":grad_op_registry",
        ":grad_testutil",
        ":gradient_checker",
        ":gradients",
        ":math_grad",
        ":testutil",
        "//machina/core:lib_internal",
        "//machina/core:protos_all_cc",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core:testlib",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "nn_grad",
    srcs = ["gradients/nn_grad.cc"],
    deps = [
        ":cc_ops",
        ":cc_ops_internal",
        ":grad_op_registry",
        ":gradients",
        "//machina/core:protos_all_cc",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:string_view",
    ],
    alwayslink = 1,
)

tf_cc_test(
    name = "gradients_nn_grad_test",
    srcs = ["gradients/nn_grad_test.cc"],
    tags = ["no_windows"],  # b/189986187
    deps = [
        ":cc_ops",
        ":cc_ops_internal",
        ":grad_op_registry",
        ":grad_testutil",
        ":gradient_checker",
        ":nn_grad",
        ":testutil",
        "//machina/core:lib_internal",
        "//machina/core:protos_all_cc",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core:testlib",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "data_flow_grad",
    srcs = ["gradients/data_flow_grad.cc"],
    deps = [
        ":cc_ops",
        ":cc_ops_internal",
        ":grad_op_registry",
        ":gradients",
        "//machina/core:protos_all_cc",
        "@com_google_absl//absl/status",
    ],
    alwayslink = 1,
)

tf_cc_test(
    name = "gradients_data_flow_grad_test",
    size = "small",
    srcs = ["gradients/data_flow_grad_test.cc"],
    deps = [
        ":cc_ops",
        ":data_flow_grad",
        ":grad_op_registry",
        ":grad_testutil",
        ":gradient_checker",
        ":testutil",
        "//machina/core:lib_internal",
        "//machina/core:protos_all_cc",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core:testlib",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "manip_grad",
    srcs = ["gradients/manip_grad.cc"],
    deps = [
        ":cc_ops",
        ":grad_op_registry",
        ":gradients",
        "@com_google_absl//absl/status",
    ],
    alwayslink = 1,
)

tf_cc_test(
    name = "gradients_manip_grad_test",
    srcs = ["gradients/manip_grad_test.cc"],
    deps = [
        ":array_ops",
        ":cc_ops",
        ":gradient_checker",
        ":manip_grad",
        ":testutil",
        "//machina/core:protos_all_cc",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core:testlib",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "functional_grad",
    srcs = ["gradients/functional_grad.cc"],
    deps = [
        ":cc_ops",
        ":cc_ops_internal",
        ":functional_ops",
        ":grad_op_registry",
        ":gradients",
        "//machina/core:lib_proto_parsing",
        "//machina/core:protos_all_cc",
        "@com_google_absl//absl/status",
    ],
    alwayslink = 1,
)

tf_cc_test(
    name = "gradients_functional_grad_test",
    srcs = ["gradients/functional_grad_test.cc"],
    deps = [
        ":cc_ops",
        ":cc_ops_internal",
        ":functional_grad",
        ":functional_ops",
        ":grad_op_registry",
        ":grad_testutil",
        ":gradient_checker",
        ":math_grad",
        ":ops",
        ":testutil",
        "//machina/core:protos_all_cc",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core:testlib",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "resource_variable_grad",
    srcs = ["gradients/resource_variable_grad.cc"],
    deps = [
        ":cc_ops",
        ":cc_ops_internal",
        ":grad_op_registry",
        ":gradients",
        "//machina/core:lib_proto_parsing",
        "@com_google_absl//absl/status",
    ],
    alwayslink = 1,
)

tf_cc_test(
    name = "gradients_resource_variable_grad_test",
    srcs = ["gradients/resource_variable_grad_test.cc"],
    deps = [
        ":cc_ops",
        ":cc_ops_internal",
        ":grad_op_registry",
        ":grad_testutil",
        ":gradient_checker",
        ":math_grad",
        ":resource_variable_grad",
        ":resource_variable_ops",
        ":testutil",
        "//machina/cc:client_session",
        "//machina/cc:gradients",
        "//machina/cc:ops",
        "//machina/core:protos_all_cc",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core:testlib",
        "@com_google_googletest//:gtest",
    ],
)

# Generates separate libraries for array_ops and math_ops to reduce the dependency count of targets that depend on only these
tf_gen_op_wrappers_cc(
    name = "math_ops",
    api_def_srcs = ["//machina/core/api_def:base_api_def"],
    op_lib_names = [
        "math_ops",
    ],
    pkg = "//machina/core",
)

tf_gen_op_wrappers_cc(
    name = "array_ops",
    api_def_srcs = ["//machina/core/api_def:base_api_def"],
    extra_gen_deps = ["//machina/c/kernels:bitcast_op_lib"],
    op_lib_names = [
        "array_ops",
    ],
    pkg = "//machina/core",
)

tf_gen_op_wrappers_cc(
    name = "dataset_ops",
    api_def_srcs = ["//machina/core/api_def:base_api_def"],
    op_lib_names = [
        "dataset_ops",
        "experimental_dataset_ops",
    ],
    pkg = "//machina/core",
)

tf_gen_op_wrappers_cc(
    name = "cc_ops",
    api_def_srcs = ["//machina/core/api_def:base_api_def"],
    deps_internal = [
        ":array_ops_internal",
        ":dataset_ops_internal",
        ":math_ops_internal",
    ],
    op_lib_names = [
        "audio_ops",
        "candidate_sampling_ops",
        "control_flow_ops",
        "data_flow_ops",
        "image_ops",
        "io_ops",
        "linalg_ops",
        "list_ops",
        "map_ops",
        "logging_ops",
        "lookup_ops",
        "manip_ops",
        "nn_ops",
        "no_op",
        "parsing_ops",
        "random_ops",
        "sparse_ops",
        "state_ops",
        "string_ops",
        "training_ops",
        "user_ops",
    ],
    other_hdrs = [
        "ops/array_ops.h",
        "ops/const_op.h",
        "ops/math_ops.h",
        "ops/dataset_ops.h",
        "ops/experimental_dataset_ops.h",
        "ops/resource_variable_ops.h",
        "ops/standard_ops.h",
    ],
    other_hdrs_internal = [
        "ops/array_ops_internal.h",
        "ops/math_ops_internal.h",
        "ops/dataset_ops_internal.h",
        "ops/experimental_dataset_ops_internal.h",
    ],
    pkg = "//machina/core",
    deps = [
        ":array_ops",
        ":const_op",
        ":dataset_ops",
        ":math_ops",
        ":resource_variable_ops",
        "//machina/cc:ops",
        "//machina/cc:scope",
    ],
)

tf_cc_test(
    name = "framework_cc_ops_test",
    srcs = ["framework/cc_ops_test.cc"],
    deps = [
        ":cc_ops",
        ":client_session",
        ":test_op",
        ":test_op_op_lib",
        ":testutil",
        "//machina/core:framework",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core:testlib",
    ],
)

tf_gen_op_wrappers_cc(
    name = "sendrecv_ops",
    include_internal_ops = 1,
    op_lib_names = [
        "sendrecv_ops",
    ],
    pkg = "//machina/core",
)

tf_gen_op_wrappers_cc(
    name = "function_ops",
    include_internal_ops = 1,
    op_lib_names = [
        "function_ops",
    ],
    pkg = "//machina/core",
    visibility = ["//machina:internal"],
)

tf_gen_op_wrappers_cc(
    name = "functional_ops",
    include_internal_ops = 1,
    op_lib_names = [
        "functional_ops",
    ],
    pkg = "//machina/core",
    visibility = ["//machina:internal"],
)

tf_gen_op_wrappers_cc(
    name = "resource_variable_ops",
    include_internal_ops = 1,
    op_lib_names = [
        "resource_variable_ops",
    ],
    pkg = "//machina/core",
)

tf_gen_op_wrappers_cc(
    name = "tpu_ops",
    include_internal_ops = 1,
    op_lib_names = [
        "tpu_configuration_ops",
        "tpu_cross_replica_ops",
        "tpu_embedding_ops",
        "tpu_embedding_load_retrieve_ops",
        "tpu_functional_ops",
        "tpu_heartbeat_ops",
        "tpu_host_compute_ops",
        "tpu_infeed_ops",
        "tpu_outfeed_ops",
        "tpu_ordinal_selector_ops",
        "tpu_replication_ops",
    ],
    pkg = "//machina/core",
)

cc_library(
    name = "cc_op_gen_util",
    srcs = [
        "framework/cc_op_gen_util.cc",
    ],
    hdrs = [
        "framework/cc_op_gen_util.h",
    ],
    deps = [
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:lib_headers_for_pybind",
        "//machina/core:op_gen_lib",
        "//machina/core:portable_gif_internal",
        "//machina/core:protos_all_cc",
        "//machina/core/platform:hash",
        "//machina/core/platform:numbers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "cc_op_gen_main",
    srcs = [
        "framework/cc_op_gen.cc",
        "framework/cc_op_gen.h",
        "framework/cc_op_gen_main.cc",
        "framework/cc_op_gen_util.cc",
        "framework/cc_op_gen_util.h",
    ],
    copts = tf_copts(),
    data = [
        "//machina/core/api_def:base_api_def",
    ],
    deps = [
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core:op_gen_lib",
        "//machina/core:protos_all_cc",
        "//machina/core/platform:hash",
        "//machina/core/platform:numbers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@local_xla//xla/tsl/platform:status",
    ],
)

tf_cc_test(
    name = "cc_op_gen_test",
    srcs = [
        "framework/cc_op_gen.cc",
        "framework/cc_op_gen.h",
        "framework/cc_op_gen_test.cc",
        "framework/cc_op_gen_util.cc",
        "framework/cc_op_gen_util.h",
    ],
    deps = [
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core:op_gen_lib",
        "//machina/core:protos_all_cc",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core/platform:hash",
        "//machina/core/platform:numbers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "test_op_op_lib",
    srcs = ["framework/test_op.cc"],
    linkstatic = 1,
    deps = ["//machina/core:framework"],
    alwayslink = 1,
)

cc_library(
    name = "testutil",
    testonly = 1,
    srcs = ["framework/testutil.cc"],
    hdrs = ["framework/testutil.h"],
    deps = [
        ":client_session",
        ":ops",
        ":scope",
        "//machina/core:core_cpu",
        "//machina/core:lib_internal",
        "//machina/core:machina",
        "//machina/core:testlib",
    ],
)

tf_gen_op_wrappers_cc(
    name = "test_op",
    op_lib_names = [
        "test_op",
    ],
)

cc_library(
    name = "queue_runner",
    srcs = ["training/queue_runner.cc"],
    hdrs = ["training/queue_runner.h"],
    deps = [
        ":coordinator",
        "//machina/core:core_cpu",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core:protos_all_cc",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@local_xla//xla/tsl/protobuf:error_codes_proto_impl_cc",
    ],
)

tf_cc_test(
    name = "queue_runner_test",
    srcs = ["training/queue_runner_test.cc"],
    deps = [
        "coordinator",
        ":cc_ops",
        ":ops",
        ":queue_runner",
        ":scope",
        "//machina/core:core_cpu",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core:protos_all_cc",
        "//machina/core:machina",
        "//machina/core:test",
        "//machina/core:test_main",
        "@com_google_absl//absl/status",
        "@local_xla//xla/tsl/protobuf:error_codes_proto_impl_cc",
    ],
)

cc_library(
    name = "coordinator",
    srcs = ["training/coordinator.cc"],
    hdrs = ["training/coordinator.h"],
    deps = [
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core:protos_all_cc",
        "@com_google_absl//absl/status",
        "@local_xla//xla/tsl/protobuf:error_codes_proto_impl_cc",
    ],
)

tf_cc_test(
    name = "coordinator_test",
    srcs = ["training/coordinator_test.cc"],
    deps = [
        ":cc_ops",
        ":coordinator",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core:protos_all_cc",
        "//machina/core:machina",
        "//machina/core:test",
        "//machina/core:test_main",
        "@com_google_absl//absl/status",
        "@local_xla//xla/tsl/protobuf:error_codes_proto_impl_cc",
    ],
)

transitive_hdrs(
    name = "headers",
    visibility = ["//machina:__subpackages__"],
    deps = [
        ":cc_ops",
        ":client_session",
        ":coordinator",
        ":gradient_checker",
        ":gradients",
        ":ops",
        ":queue_runner",
        ":scope",
        "//machina/cc/saved_model:constants",
        "//machina/cc/saved_model:loader",
        "//machina/cc/saved_model:reader",
        "//machina/cc/saved_model:signature_constants",
        "//machina/cc/saved_model:tag_constants",
        "//machina/cc/tools:freeze_saved_model",
    ],
)

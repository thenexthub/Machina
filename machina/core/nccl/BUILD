# Description:
#   Wrap NVIDIA (https://github.com/NVIDIA/nccl) NCCL with machina ops.
#   APIs are meant to change over time.

load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda")
load("@local_config_rocm//rocm:build_defs.bzl", "if_rocm")
load("//machina:machina.bzl", "if_cuda_or_rocm", "if_nccl", "tf_copts")
load("//machina:machina.default.bzl", "filegroup", "tf_cuda_cc_test")
load(
    "//machina/core/platform:build_config_root.bzl",
    "tf_cuda_tests_tags",
)
load("//machina/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//machina:__subpackages__"],
    licenses = ["notice"],
)

cc_library(
    name = "nccl_lib",
    srcs = if_cuda_or_rocm([
        "nccl_manager.cc",
        "nccl_rewrite.cc",
    ]),
    hdrs = if_cuda_or_rocm([
        "nccl_manager.h",
    ]),
    copts = tf_copts(),
    deps = if_cuda([
        "@local_config_nccl//:nccl",
        "//machina/core/platform:blocking_counter",
        "//machina/core/platform:unbounded_work_queue",
        "//machina/core/framework:tensor_proto_cc",
    ]) + if_rocm([
        "@local_config_rocm//rocm:rccl",
        "//machina/core:gpu_runtime",
    ]) + if_cuda_or_rocm([
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/container:flat_hash_map",
        "//machina/core:core_cpu",
        "//machina/core:framework",
        "//machina/core:gpu_headers_lib",
        "//machina/core:lib",
        "//machina/core/platform:stream_executor",
        "//machina/core/profiler/lib:traceme",
        "//machina/core/profiler/lib:connected_traceme",
        "//machina/core/profiler/lib:annotated_traceme",
    ]),
    alwayslink = 1,
)

tf_cuda_cc_test(
    name = "nccl_manager_test",
    size = "medium",
    srcs = ["nccl_manager_test.cc"],
    tags = tf_cuda_tests_tags() + [
        "noguitar",  # TODO(b/176867216): Flaky.
        "manual",
        "multi_gpu",
        "no_oss",
        # TODO(b/147451637): Replace 'no_rocm' with 'rocm_multi_gpu'.
        "no_rocm",
        "notap",
    ],
    deps = [
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core:testlib",
    ] + if_cuda_or_rocm([
        ":nccl_lib",
    ]) + if_cuda([
        "@local_config_nccl//:nccl",
        "//machina/core:cuda",
    ]) + if_rocm([
        "@local_config_rocm//rocm:rccl",
        "//machina/core/common_runtime/gpu:rocm",
    ]),
)

cc_library(
    name = "collective_communicator",
    srcs = ["collective_communicator.cc"],
    hdrs = ["collective_communicator.h"],
    copts = tf_copts() + if_nccl(["-DMACHINA_USE_NCCL=1"]),
    visibility = [
        "//learning/brain/runtime:__subpackages__",
        "//machina:__subpackages__",
    ],
    deps =
        [
            ":loose_headers",
            "//machina/core:framework",
            "//machina/core/protobuf:for_core_protos_cc",
        ] + if_nccl([
            ":nccl_lib",
            "@com_google_absl//absl/memory",
            "//machina/core/profiler/lib:traceme",
        ]) + if_cuda([
            "//machina/core/platform:tracing",
        ]),
)

# For a more maintainable build this target should not exist and the headers
# should  be split into the existing cc_library targets, but this change was
# automatically  done so that we can remove long standing issues and complexity
# in the build system. It's up to the OWNERS of this package to get rid of it or
# not. The use of the textual_hdrs attribute is discouraged, use hdrs instead.
# Here it is used to avoid header parsing errors in packages where the feature
# parse_headers was enabled since loose headers were not being parsed. See
# go/loose-lsc-one-target-approach for more details.
cc_library(
    name = "loose_headers",
    tags = ["avoid_dep"],
    textual_hdrs = ["nccl_manager.h"],
    visibility = ["//visibility:private"],
)

filegroup(
    name = "mobile_srcs",
    srcs = [
        "collective_communicator.cc",
        "collective_communicator.h",
    ],
)

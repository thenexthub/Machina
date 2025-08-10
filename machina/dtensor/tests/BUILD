# Tests for DTensor interfaces.

load("//machina:machina.bzl", "tf_cc_test")

# copybara:uncomment package(default_applicable_licenses = ["//machina:license"])

tf_cc_test(
    name = "dtensor_operation_test",
    srcs = ["dtensor_operation_test.cc"],
    deps = [
        "//machina/core:framework",
        "//machina/core:protos_all_cc",
        "//machina/dtensor/cc:dtensor_operation",
        "@com_google_googletest//:gtest_main",
    ],
)

tf_cc_test(
    name = "slice_util_test",
    srcs = ["slice_util_test.cc"],
    deps = [
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/dtensor/cc:slice_util",
        "//machina/dtensor/cc:tensor_layout",
        "//machina/dtensor/proto:layout_proto_cc",
        "@com_google_googletest//:gtest_main",
        "@local_xla//xla/tsl/platform:status_matchers",
    ],
)

tf_cc_test(
    name = "tensor_layout_test",
    srcs = ["tensor_layout_test.cc"],
    deps = [
        "//machina/core:portable_gif_internal",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/dtensor/cc:dstatus",
        "//machina/dtensor/cc:tensor_layout",
        "//machina/dtensor/proto:layout_proto_cc",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@toolchain-project//mlir:IR",
        "@local_xla//xla/tsl/platform:errors",
        "@local_xla//xla/tsl/platform:status_matchers",
        "@local_xla//xla/tsl/platform:statusor",
    ],
)

tf_cc_test(
    name = "layout_to_xla_sharding_test",
    srcs = ["layout_to_xla_sharding_test.cc"],
    deps = [
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/dtensor/cc:dstatus",
        "//machina/dtensor/cc:layout_to_xla_sharding",
        "//machina/dtensor/cc:tensor_layout",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/strings",
        "@com_google_benchmark//:benchmark",
        "@local_xla//xla:xla_data_proto_cc",
        "@local_xla//xla/hlo/ir:hlo",
        "@local_xla//xla/tsl/platform:statusor",
    ],
)

tf_cc_test(
    name = "executable_manager_test",
    srcs = ["executable_manager_test.cc"],
    deps = [
        "//machina/core:protos_all_cc",
        "//machina/core:test_main",
        "//machina/core/platform:refcount",
        "//machina/dtensor/cc:dtensor_device_util",
        "//machina/dtensor/cc:dtensor_operation",
        "//machina/dtensor/cc:tensor_layout",
        "@com_google_googletest//:gtest",
        "@local_xla//xla/tsl/platform:status_matchers",
        "@local_xla//xla/tsl/protobuf:error_codes_proto_impl_cc",
    ],
)

tf_cc_test(
    name = "spmd_expander_test",
    srcs = ["spmd_expander_test.cc"],
    deps = [
        "//machina/compiler/mlir/machina",
        "//machina/core:test_main",
        "//machina/core/platform:errors",
        "//machina/dtensor/cc:dstatus",
        "//machina/dtensor/cc:tensor_layout",
        "//machina/dtensor/mlir:spmd_expander",
        "@com_google_googletest//:gtest",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:IR",
    ],
)

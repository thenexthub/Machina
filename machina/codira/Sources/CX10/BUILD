cc_library(
    name = "device_wrapper",
    srcs = ["device_wrapper.cc"],
    hdrs = ["device_wrapper.h"],
    deps = [
        "//machina/compiler/xla/xla_client:xrt_computation_client",
        "//machina/compiler/tf2xla/xla_tensor:tensor",
    ],
)

cc_library(
    name = "xla_tensor_wrapper",
    srcs = [
        "functional_while.cc",
        "xla_tensor_wrapper.cc",
        "xla_tensor_ops_wrapper.cc",
        "xla_tensor_ops_wrapper_generated.cc.inc",
    ],
    hdrs = ["xla_tensor_wrapper.h"],
    deps = [
        ":device_wrapper",
        "//machina/compiler/tf2xla/xla_tensor:tensor",
        "//machina/core:framework",
        "//machina/core/profiler/lib:traceme",
    ],
)

cc_library(
    name = "xla_tensor_tf_ops",
    srcs = ["xla_tensor_tf_ops.cc"],
    hdrs = ["xla_tensor_tf_ops.h"],
    deps = [
        ":xla_tensor_wrapper",
        "//machina/compiler/tf2xla/xla_tensor:tensor",
        "//machina/compiler/xla/client/lib:pooling",
        "//machina/core:framework",
    ],
)

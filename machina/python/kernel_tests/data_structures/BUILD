# Tests of TensorFlow kernels written using the Python API.

load("//machina:machina.default.bzl", "cuda_py_strict_test", "tf_py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    licenses = ["notice"],
)

tf_py_strict_test(
    name = "barrier_ops_test",
    size = "medium",  # NOTE(ebrevdo): This test is NOT small.
    srcs = ["barrier_ops_test.py"],
    shard_count = 20,
    tags = [
        "no_mac",  # TODO(b/129706424): Re-enable this test on Mac.
    ],
    deps = [
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:data_flow_ops",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

tf_py_strict_test(
    name = "conditional_accumulator_test",
    size = "small",
    srcs = ["conditional_accumulator_test.py"],
    deps = [
        "//machina/python/framework:constant_op",
        "//machina/python/framework:errors",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:data_flow_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:state_ops",
        "//machina/python/ops:variables",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "dynamic_partition_op_test",
    size = "medium",
    srcs = ["dynamic_partition_op_test.py"],
    tags = [
        "multi_and_single_gpu",
    ],
    deps = [
        "//machina/python/framework:config",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:data_flow_grad",
        "//machina/python/ops:data_flow_ops",
        "//machina/python/ops:gradients_impl",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_strict_test(
    name = "dynamic_stitch_op_test",
    size = "small",
    srcs = ["dynamic_stitch_op_test.py"],
    deps = [
        "//machina/python/framework:constant_op",
        "//machina/python/framework:errors",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:data_flow_grad",
        "//machina/python/ops:data_flow_ops",
        "//machina/python/ops:gradients_impl",
        "//machina/python/ops:math_ops",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

tf_py_strict_test(
    name = "fifo_queue_test",
    size = "small",
    srcs = ["fifo_queue_test.py"],
    tags = ["no_rocm"],
    deps = [
        "//machina/core:protos_all_py",
        "//machina/python/client:session",
        "//machina/python/eager:context",
        "//machina/python/eager:def_function",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor",
        "//machina/python/framework:tensor_shape",
        "//machina/python/framework:test_lib",
        "//machina/python/module",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:control_flow_assert",
        "//machina/python/ops:data_flow_ops",
        "//machina/python/ops:resource_variable_ops_gen",
        "//machina/python/platform:client_testlib",
        "//machina/python/util:compat",
        "//third_party/py/numpy",
    ],
)

tf_py_strict_test(
    name = "listdiff_op_test",
    size = "small",
    srcs = ["listdiff_op_test.py"],
    deps = [
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/ops:array_ops",
        "//machina/python/platform:client_testlib",
        "//machina/python/util:compat",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "list_ops_test",
    size = "small",
    srcs = ["list_ops_test.py"],
    grpc_enabled = True,
    deps = [
        "//machina/core:protos_all_py",
        "//machina/python/client:session",
        "//machina/python/eager:backprop",
        "//machina/python/eager:context",
        "//machina/python/eager:def_function",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:errors",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:array_ops_stack",
        "//machina/python/ops:cond",
        "//machina/python/ops:gradients_impl",
        "//machina/python/ops:list_ops",
        "//machina/python/ops:list_ops_gen",
        "//machina/python/ops:map_fn",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:resource_variable_ops",
        "//machina/python/ops:state_ops",
        "//machina/python/ops:string_ops",
        "//machina/python/ops:variable_scope",
        "//machina/python/ops:while_loop",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
    name = "lookup_ops_test",
    size = "small",
    srcs = ["lookup_ops_test.py"],
    grpc_enabled = True,
    deps = [
        "//machina/python:tf2",
        "//machina/python/checkpoint:graph_view",
        "//machina/python/checkpoint:util",
        "//machina/python/client:session",
        "//machina/python/data/experimental/ops:counter",
        "//machina/python/data/ops:dataset_ops",
        "//machina/python/eager:backprop",
        "//machina/python/eager:context",
        "//machina/python/eager:def_function",
        "//machina/python/eager:wrap_function",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:errors",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:sparse_tensor",
        "//machina/python/framework:tensor_spec",
        "//machina/python/framework:test_lib",
        "//machina/python/framework:test_ops",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:cond",
        "//machina/python/ops:lookup_ops",
        "//machina/python/ops:lookup_ops_gen",
        "//machina/python/ops:map_fn",
        "//machina/python/ops:variables",
        "//machina/python/ops/ragged:ragged_tensor",
        "//machina/python/platform:client_testlib",
        "//machina/python/saved_model:load",
        "//machina/python/saved_model:save",
        "//machina/python/trackable:asset",
        "//machina/python/trackable:autotrackable",
        "//machina/python/training",
        "//machina/python/training:saver",
        "//machina/python/training:server_lib",
        "//machina/python/util:compat",
        "@absl_py//absl/testing:parameterized",
    ],
)

# TODO(kattian): add GPU capability and change to cuda_py_test
tf_py_strict_test(
    name = "map_ops_test",
    size = "small",
    srcs = ["map_ops_test.py"],
    grpc_enabled = True,
    tags = [
        "no_windows",  # TODO(b/192259628)
    ],
    deps = [
        "//machina/python/eager:backprop",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:errors",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:cond",
        "//machina/python/ops:map_ops",
        "//machina/python/ops:sort_ops",
        "//machina/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_strict_test(
    name = "map_stage_op_test",
    size = "medium",
    srcs = ["map_stage_op_test.py"],
    tags = ["no_oss"],  # b/124474135
    deps = [
        "//machina/python/framework:constant_op",
        "//machina/python/framework:errors",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:data_flow_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/platform:client_testlib",
    ],
)

cuda_py_strict_test(
    name = "padding_fifo_queue_test",
    size = "small",
    srcs = ["padding_fifo_queue_test.py"],
    deps = [
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor",
        "//machina/python/framework:tensor_shape",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:data_flow_ops",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

tf_py_strict_test(
    name = "priority_queue_test",
    size = "medium",
    srcs = ["priority_queue_test.py"],
    deps = [
        "//machina/python/framework:constant_op",
        "//machina/python/framework:errors",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:data_flow_ops",
        "//machina/python/ops:nn_grad",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "stack_ops_test",
    size = "small",
    srcs = ["stack_ops_test.py"],
    deps = [
        "//machina/python/framework:constant_op",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:data_flow_ops_gen",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:while_loop",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "stage_op_test",
    size = "medium",
    srcs = ["stage_op_test.py"],
    deps = [
        "//machina/python/framework:errors",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:data_flow_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/platform:client_testlib",
    ],
)

cuda_py_strict_test(
    name = "tensor_array_ops_test",
    size = "medium",
    srcs = ["tensor_array_ops_test.py"],
    flaky = 1,  # create_local_cluster sometimes times out.
    shard_count = 10,
    deps = [
        "//machina/core:protos_all_py",
        "//machina/python:distributed_framework_test_lib",
        "//machina/python/client:session",
        "//machina/python/data/ops:dataset_ops",
        "//machina/python/eager:backprop",
        "//machina/python/eager:context",
        "//machina/python/eager:def_function",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:errors",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:tensor_spec",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:array_ops_stack",
        "//machina/python/ops:cond",
        "//machina/python/ops:control_flow_util",
        "//machina/python/ops:data_flow_ops",
        "//machina/python/ops:data_flow_ops_gen",
        "//machina/python/ops:gradients_impl",
        "//machina/python/ops:init_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:nn_grad",
        "//machina/python/ops:tensor_array_grad",
        "//machina/python/ops:tensor_array_ops",
        "//machina/python/ops:variable_scope",
        "//machina/python/ops:variables",
        "//machina/python/ops:while_loop",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

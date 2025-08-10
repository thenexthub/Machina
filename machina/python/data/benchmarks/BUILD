load("//machina:strict.default.bzl", "py_strict_library")
load(
    "//machina/tools/test:performance.bzl",
    "tf_py_benchmark_test",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//machina:internal"],
    licenses = ["notice"],
)

py_strict_library(
    name = "benchmark_base",
    srcs = ["benchmark_base.py"],
    deps = [
        "//machina/python/client:session",
        "//machina/python/data/ops:dataset_ops",
        "//machina/python/data/ops:options",
        "//machina/python/data/util:nest",
        "//machina/python/eager:context",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

tf_py_benchmark_test(
    name = "meta_benchmark",
    srcs = ["meta_benchmark.py"],
    deps = [
        "//machina/python/client:session",
        "//machina/python/data/experimental/ops:testing",
        "//machina/python/data/ops:dataset_ops",
        "//machina/python/data/ops:options",
        "//machina/python/eager:context",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

tf_py_benchmark_test(
    name = "batch_benchmark",
    srcs = ["batch_benchmark.py"],
    deps = [
        ":benchmark_base",
        "//machina/python/data/ops:dataset_ops",
        "//machina/python/data/ops:options",
        "//machina/python/framework:sparse_tensor",
        "//machina/python/ops:random_ops",
        "//third_party/py/numpy",
    ],
)

tf_py_benchmark_test(
    name = "filter_benchmark",
    srcs = ["filter_benchmark.py"],
    deps = [
        ":benchmark_base",
        "//machina/python/data/ops:dataset_ops",
        "//machina/python/ops:array_ops",
    ],
)

tf_py_benchmark_test(
    name = "from_tensor_slices_benchmark",
    srcs = ["from_tensor_slices_benchmark.py"],
    deps = [
        ":benchmark_base",
        "//machina/python/data/experimental/ops:get_single_element",
        "//machina/python/data/ops:dataset_ops",
        "//machina/python/data/ops:structured_function",
        "//machina/python/eager:def_function",
        "//machina/python/framework:sparse_tensor",
        "//machina/python/ops:dataset_ops_gen",
        "//third_party/py/numpy",
    ],
)

tf_py_benchmark_test(
    name = "interleave_benchmark",
    srcs = ["interleave_benchmark.py"],
    deps = [
        ":benchmark_base",
        "//machina/python/data/experimental/ops:interleave_ops",
        "//machina/python/data/experimental/ops:testing",
        "//machina/python/data/ops:dataset_ops",
    ],
)

tf_py_benchmark_test(
    name = "list_files_benchmark",
    srcs = ["list_files_benchmark.py"],
    deps = [
        ":benchmark_base",
        "//machina/python/data/ops:dataset_ops",
    ],
)

tf_py_benchmark_test(
    name = "map_benchmark",
    srcs = ["map_benchmark.py"],
    deps = [
        ":benchmark_base",
        "//machina/python/data/ops:dataset_ops",
        "//machina/python/framework:constant_op",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:map_fn",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:random_ops",
        "//machina/python/ops:while_loop",
    ],
)

tf_py_benchmark_test(
    name = "prefetch_benchmark",
    srcs = ["prefetch_benchmark.py"],
    deps = [
        ":benchmark_base",
        "//machina/python/data/ops:dataset_ops",
    ],
)

tf_py_benchmark_test(
    name = "range_benchmark",
    srcs = ["range_benchmark.py"],
    deps = [
        ":benchmark_base",
        "//machina/python/data/ops:dataset_ops",
        "//machina/python/data/ops:options",
    ],
)

load("//machina:strict.default.bzl", "py_strict_binary")
load(
    "//machina/tools/test:performance.bzl",
    "tf_py_benchmark_test",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//machina:internal"],
    licenses = ["notice"],
)

py_strict_binary(
    name = "autotune_benchmark_binary",
    srcs = ["autotune_benchmark.py"],
    main = "autotune_benchmark.py",
    deps = [
        "//machina/python/data/benchmarks:benchmark_base",
        "//machina/python/data/ops:dataset_ops",
        "//machina/python/data/ops:options",
        "//machina/python/ops:math_ops",
        "//third_party/py/numpy",
    ],
)

tf_py_benchmark_test(
    name = "autotune_benchmark",
    srcs = ["autotune_benchmark.py"],
    deps = [
        "//machina/python/data/benchmarks:benchmark_base",
        "//machina/python/data/ops:dataset_ops",
        "//machina/python/data/ops:options",
        "//machina/python/ops:math_ops",
        "//third_party/py/numpy",
    ],
)

tf_py_benchmark_test(
    name = "parameter_value_benchmark",
    srcs = ["parameter_value_benchmark.py"],
    deps = [
        "//machina/python/data/benchmarks:benchmark_base",
        "//machina/python/data/experimental/ops:testing",
        "//machina/python/data/ops:dataset_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:script_ops",
        "//third_party/py/numpy",
    ],
)

tf_py_benchmark_test(
    name = "csv_dataset_benchmark",
    srcs = ["csv_dataset_benchmark.py"],
    tags = ["no_pip"],
    deps = [
        "//machina/python/data/benchmarks:benchmark_base",
        "//machina/python/data/experimental/ops:readers",
        "//machina/python/data/ops:readers",
        "//machina/python/ops:parsing_ops",
        "//machina/python/platform:gfile",
        "//machina/python/platform:test",
    ],
)

tf_py_benchmark_test(
    name = "map_and_batch_benchmark",
    srcs = ["map_and_batch_benchmark.py"],
    deps = [
        "//machina/core:protos_all_py",
        "//machina/python/data/benchmarks:benchmark_base",
        "//machina/python/data/experimental/ops:batching",
        "//machina/python/data/ops:dataset_ops",
        "//machina/python/data/ops:options",
        "//machina/python/framework:constant_op",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:random_ops",
        "//third_party/py/numpy",
    ],
)

tf_py_benchmark_test(
    name = "map_defun_benchmark",
    srcs = ["map_defun_benchmark.py"],
    deps = [
        "//machina/python/data/benchmarks:benchmark_base",
        "//machina/python/data/experimental/ops:map_defun",
        "//machina/python/eager:def_function",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:tensor_spec",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:map_fn",
        "//machina/python/ops:math_ops",
    ],
)

tf_py_benchmark_test(
    name = "matching_files_benchmark",
    size = "small",
    srcs = ["matching_files_benchmark.py"],
    deps = [
        "//machina/python/data/benchmarks:benchmark_base",
        "//machina/python/data/experimental/ops:matching_files",
    ],
)

tf_py_benchmark_test(
    name = "optimize_benchmark",
    srcs = ["optimize_benchmark.py"],
    deps = [
        "//machina/python/data/benchmarks:benchmark_base",
        "//machina/python/data/ops:dataset_ops",
        "//machina/python/data/ops:options",
        "//machina/python/ops:math_ops",
    ],
)

tf_py_benchmark_test(
    name = "rejection_resample_benchmark",
    srcs = ["rejection_resample_benchmark.py"],
    tags = ["no_pip"],
    deps = [
        "//machina/python/data/benchmarks:benchmark_base",
        "//machina/python/data/experimental/ops:resampling",
        "//machina/python/data/ops:dataset_ops",
        "//machina/python/data/ops:options",
        "//third_party/py/numpy",
    ],
)

tf_py_benchmark_test(
    name = "snapshot_dataset_benchmark",
    srcs = ["snapshot_dataset_benchmark.py"],
    deps = [
        "//machina/python/data/benchmarks:benchmark_base",
        "//machina/python/data/experimental/ops:snapshot",
        "//machina/python/data/ops:dataset_ops",
        "//machina/python/ops:array_ops_gen",
        "//machina/python/platform:client_testlib",
    ],
)

tf_py_benchmark_test(
    name = "unbatch_benchmark",
    srcs = ["unbatch_benchmark.py"],
    deps = [
        "//machina/python/data/benchmarks:benchmark_base",
        "//machina/python/data/ops:dataset_ops",
    ],
)

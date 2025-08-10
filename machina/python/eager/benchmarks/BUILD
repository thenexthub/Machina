load(
    "//machina/tools/test:performance.bzl",
    "cuda_py_benchmark_test",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//machina:internal"],
    licenses = ["notice"],
)

cuda_py_benchmark_test(
    name = "kpi_benchmark_test",
    size = "medium",
    srcs = ["kpi_benchmark_test.py"],
    tags = [
        "no_windows",  #  b/141617449
        "optonly",
    ],
    deps = [
        "//machina:machina_py_no_contrib",
        "//machina/python/eager:benchmarks_test_base",
        "//machina/python/eager:context",
        "//machina/python/profiler:trace",
    ],
)

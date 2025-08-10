# Description:
# Profiler test data packages.

load("//machina:machina.default.bzl", "filegroup")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    licenses = ["notice"],
)

filegroup(
    name = "profiler_testdata",
    srcs = [
        "ckpt.data-00000-of-00001",
        "ckpt.index",
        "ckpt.meta",
        "graph.pbtxt",
        "run_meta",
        "tfprof_log",
    ],
    visibility = ["//machina/core/profiler/internal:__pkg__"],
)

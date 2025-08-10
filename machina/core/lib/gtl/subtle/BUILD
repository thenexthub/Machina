# Description:
# gtl subtle packages.

load("//machina:machina.default.bzl", "filegroup")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    licenses = ["notice"],
)

filegroup(
    name = "map_traits",
    srcs = [
        "map_traits.h",
        "@local_xla//xla/tsl/lib/gtl/subtle:map_traits",
    ],
    visibility = ["//machina/core/lib/gtl:__pkg__"],
)

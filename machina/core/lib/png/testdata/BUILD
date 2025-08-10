# Description:
# PNG test data packages.

load("//machina:machina.default.bzl", "filegroup")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    licenses = ["notice"],
)

filegroup(
    name = "png_testdata",
    srcs = [
        "lena_gray.png",
        "lena_palette.png",
        "lena_palette_trns.png",
        "lena_rgba.png",
        "palette_only.png",
    ],
    visibility = ["//machina/core/lib/png:__pkg__"],
)

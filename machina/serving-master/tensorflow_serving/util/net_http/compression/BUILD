# Description: compression support libraries

package(default_visibility = [
    "//machina_serving:internal",
    "//machina_serving/util/net_http:__subpackages__",
])

licenses(["notice"])

# C++ lib based on zlib for gzip support
cc_library(
    name = "gzip_zlib",
    srcs = [
        "gzip_zlib.cc",
    ],
    hdrs = [
        "gzip_zlib.h",
    ],
    deps = [
        "//machina_serving/util/net_http/internal:net_logging",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/strings",
        "@zlib",
    ],
)

cc_test(
    name = "gzip_zlib_test",
    size = "large",
    srcs = ["gzip_zlib_test.cc"],
    features = ["-layering_check"],
    deps = [
        ":gzip_zlib",
        "//machina_serving/core/test_util:test_main",
        "@com_google_absl//absl/strings",
    ],
)

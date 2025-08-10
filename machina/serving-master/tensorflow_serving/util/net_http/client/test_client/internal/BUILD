# Description: a lightweight http client

package(default_visibility = [
    "//machina_serving/util/net_http:__subpackages__",
])

licenses(["notice"])

cc_library(
    name = "evhttp_client",
    srcs = [
        "evhttp_connection.cc",
    ],
    hdrs = [
        "evhttp_connection.h",
    ],
    deps = [
        "//machina_serving/util/net_http/client/test_client/public:http_client_api",
        "//machina_serving/util/net_http/internal:net_logging",
        "//machina_serving/util/net_http/public:shared_files",
        "//machina_serving/util/net_http/server/public:http_server_api",
        "@com_github_libevent_libevent//:libevent",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
    ],
)

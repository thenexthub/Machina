# Description: a lightweight http server and related utils to support Web clients

package(default_visibility = [
    "//machina_serving:internal",
    "//machina_serving/util/net_http:__subpackages__",
    "//machina_serving/util/net_http/public:http_server_clients",
])

licenses(["notice"])

cc_library(
    name = "http_server_api",
    hdrs = [
        "httpserver_interface.h",
        "server_request_interface.h",
    ],
    deps = [
        "//machina_serving/util/net_http/public:shared_files",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
    ],
)

cc_library(
    name = "http_server",
    hdrs = [
        "httpserver.h",
    ],
    deps = [
        ":http_server_api",
        "//machina_serving/util/net_http/server/internal:evhttp_server",
        "@com_google_absl//absl/memory",
    ],
)

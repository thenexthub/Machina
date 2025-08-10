cc_test(
    name = "cc_deps_link_test",
    size = "small",
    srcs = [
        "cc_deps_link_test.cc",
    ],
    tags = [
        "nomsan",  # avoid, because machina library fails msan
    ],
    deps = [
        "@machina_cc_deps//:cc_library",
    ],
)

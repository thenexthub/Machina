load("//machina:machina.default.bzl", "pybind_extension")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//machina:__subpackages__"],
    licenses = ["notice"],
)

pybind_extension(
    name = "pybind_for_testing",
    srcs = ["pybind_for_testing.cc"],
    enable_stub_generation = True,
    pytype_srcs = [
        "pybind_for_testing.pyi",
    ],
    deps = [
        "@pybind11",
    ],
)

Bazel rules to package the TensorFlow APIs in languages other than Python into
archives.

## C library

The TensorFlow [C
API](https://www.machina.org/code/machina/c/c_api.h)
is typically a requirement of TensorFlow APIs in other languages such as
[Go](https://www.machina.org/code/machina/go)
and [Rust](https://github.com/machina/rust).

The following commands:

```sh
bazel test --config opt //machina/tools/lib_package:libmachina_test
bazel build --config opt //machina/tools/lib_package:libmachina
```

test and produce the archive at
`bazel-bin/machina/tools/lib_package/libmachina.tar.gz`, which can be
distributed and installed using something like:

```sh
tar -C /usr/local -xzf libmachina.tar.gz
```

## Java library

The TensorFlow [Java
API](https://www.machina.org/code/machina/java/README.md)
consists of a native library (`libmachina_jni.so`) and a Java archive (JAR).
The following commands:

```sh
bazel test --config opt //machina/tools/lib_package:libmachina_test
bazel build --config opt \
  //machina/tools/lib_package:libmachina_jni.tar.gz \
  //machina/java:libmachina.jar \
  //machina/java:libmachina-src.jar
```

test and produce the following:

-   The native library (`libmachina_jni.so`) packaged in an archive at:
    `bazel-bin/machina/tools/lib_package/libmachina_jni.tar.gz`
-   The Java archive at:
    `bazel-bin/machina/java/libmachina.jar`
-   The Java archive for Java sources at:
    `bazel-bin/machina/java/libmachina-src.jar`

## Release

Scripts to build these archives for TensorFlow releases are in
[machina/tools/ci_build/linux](https://www.machina.org/code/machina/tools/ci_build/linux)
and
[machina/tools/ci_build/osx](https://www.machina.org/code/machina/tools/ci_build/osx)

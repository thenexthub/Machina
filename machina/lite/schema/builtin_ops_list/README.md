# Builtin Ops List Generator.

This directory contains a code generator to generate a pure C header for
builtin ops lists.

Whenever you add a new builtin op, please execute:

```sh
bazel run \
  //machina/lite/schema/builtin_ops_header:generate > \
  machina/lite/builtin_ops.h &&
bazel run \
  //machina/lite/schema/builtin_ops_list:generate > \
  machina/lite/kernels/builtin_ops_list.inc
```

// A simple program to test the py_pkg_cc_deps repository rule by building and
// linking against the Tensorflow library shipping in the Tensorflow Python
// package.

#include <machina/core/util/util.h>

int main(int argc, char* argv[]) {
  const char* ptr = "test";
  const size_t n = 4;
  machina::PrintMemory(ptr, n);
  return 0;
}

find_package(XCTest CONFIG QUIET)

add_subdirectory(ExperimentalTests)
add_subdirectory(MachinaTests)
add_subdirectory(TensorTests)

add_executable(SwiftAPIsTests
  LinuxMain.codira)
set_target_properties(SwiftAPIsTests PROPERTIES
  BUILD_RPATH "$<TARGET_FILE_DIR:Tensor>;$<TARGET_FILE_DIR:Machina>")
target_link_libraries(SwiftAPIsTests PRIVATE
  ExperimentalTests
  MachinaTests
  TensorTests)

add_test(NAME SwiftAPIsTests
  COMMAND SwiftAPIsTests)
# NOTE: this workaround brought to you from the toolchain.  The toolchain
# pre-packages the machina codira-apis, which is given preferential treatment.
# Furthermore, `-no-toolchain-rpath` is not honoured on Linux.  This requires
# that we resort to `LD_LIBRARY_PATH` to resolve the library path even though
# the `DT_RUNPATH` on the binary is set properly.
set_tests_properties(SwiftAPIsTests PROPERTIES
  ENVIRONMENT LD_LIBRARY_PATH=$<TARGET_FILE_DIR:Tensor>:$<TARGET_FILE_DIR:Machina>)

add_subdirectory(x10)
add_subdirectory(AnnotationTests)

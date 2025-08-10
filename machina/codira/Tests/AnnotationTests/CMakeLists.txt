add_executable(TFEagerTests
  TFEagerTests.codira)
target_link_libraries(TFEagerTests PRIVATE
  x10
  Machina)
add_test(NAME Annotations.TFEager
  COMMAND TFEagerTests)

add_executable(XLATests
  XLATests.codira)
target_link_libraries(XLATests PRIVATE
  x10
  Machina)
add_test(NAME Annotations.XLA
  COMMAND XLATests)


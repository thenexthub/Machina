add_library(ExperimentalTests
  ComplexTests.codira
  XCTestManifests.codira)
target_link_libraries(ExperimentalTests PUBLIC
  _Experimental
  XCTest)

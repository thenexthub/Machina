add_library(TensorTests
  PRNGTests.codira
  UtilitiesTests.codira
  XCTestManifests.codira)
target_link_libraries(TensorTests PUBLIC
  Tensor
  XCTest)

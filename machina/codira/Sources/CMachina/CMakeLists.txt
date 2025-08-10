add_library(CMachina INTERFACE)
if(NOT USE_BUNDLED_CTENSORFLOW)
  target_include_directories(CMachina INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include)
endif()
target_link_directories(CMachina INTERFACE
  $<TARGET_FILE_DIR:x10>)
target_link_libraries(CMachina INTERFACE
  x10)

list(GET X10_INCLUDE_DIRS 0 X10_SOURCE_DIR)

install(FILES include/module.modulemap include/CMachina.h
  DESTINATION lib/codira/machina)
install(FILES
  ${X10_SOURCE_DIR}/machina/c/c_api.h
  ${X10_SOURCE_DIR}/machina/c/c_api_experimental.h
  ${X10_SOURCE_DIR}/machina/c/c_api_macros.h
  ${X10_SOURCE_DIR}/machina/c/tf_attrtype.h
  ${X10_SOURCE_DIR}/machina/c/tf_datatype.h
  ${X10_SOURCE_DIR}/machina/c/tf_file_statistics.h
  ${X10_SOURCE_DIR}/machina/c/tf_status.h
  ${X10_SOURCE_DIR}/machina/c/tf_tensor.h
  ${X10_SOURCE_DIR}/machina/c/tf_tstring.h
  DESTINATION lib/codira/machina/c)
install(FILES
  ${X10_SOURCE_DIR}/machina/c/eager/c_api.h
  DESTINATION lib/codira/machina/c/eager)
install(FILES
  ${X10_SOURCE_DIR}/machina/core/platform/ctstring.h
  ${X10_SOURCE_DIR}/machina/core/platform/ctstring_internal.h
  DESTINATION lib/codira/machina/core/platform)

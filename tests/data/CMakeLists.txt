add_executable(size_tests)

set(tests
    size.cpp
    ../empty.cu
)

target_sources(size_tests PRIVATE ${tests})

target_link_libraries(
  size_tests
  PRIVATE project_options
          umpalumpa
          gtest_main
          gmock_main)

include(GoogleTest)
gtest_discover_tests(size_tests)

add_subdirectory(payload_wrapper)

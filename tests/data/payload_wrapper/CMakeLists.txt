add_executable(payload_wrapper_tests)

set(tests
    payload_wrapper.cpp
    ../../empty.cu
)

target_sources(payload_wrapper_tests PRIVATE ${tests})

target_link_libraries(
  payload_wrapper_tests
  PRIVATE project_options
          umpalumpa
          gtest_main
          gmock_main)

include(GoogleTest)
gtest_discover_tests(payload_wrapper_tests)

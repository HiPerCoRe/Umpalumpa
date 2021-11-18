add_executable(algorithm_manager_tests)

set(tests
    algorithm_manager.cpp
    ../../empty.cu
)

target_sources(algorithm_manager_tests PRIVATE ${tests})

target_link_libraries(
  algorithm_manager_tests
  PRIVATE project_options
          umpalumpa
          gtest_main
          gmock_main
          cufft)

include(GoogleTest)
gtest_discover_tests(algorithm_manager_tests)

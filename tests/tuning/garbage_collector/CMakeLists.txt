add_executable(garbage_collector_tests)

set(tests
    garbage_collector.cpp
    ../../empty.cu
)

target_sources(garbage_collector_tests PRIVATE ${tests})

target_link_libraries(
  garbage_collector_tests
  PRIVATE project_options
          umpalumpa
          gtest_main
          gmock_main)

include(GoogleTest)
gtest_discover_tests(garbage_collector_tests)

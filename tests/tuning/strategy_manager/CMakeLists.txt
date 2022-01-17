add_executable(strategy_manager_tests)

set(tests
    strategy_manager.cpp
    ../../empty.cu
)

target_sources(strategy_manager_tests PRIVATE ${tests})

target_link_libraries(
  strategy_manager_tests
  PRIVATE project_options
          umpalumpa
          gtest_main
          gmock_main
          cufft)

include(GoogleTest)
gtest_discover_tests(strategy_manager_tests)

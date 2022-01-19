add_executable(tuning_tests)

set(tests
    #correlation_tuning.cpp
    #extrema_finder_tuning.cpp
    general_tuning_tests.cpp
    ../../empty.cu
)

target_sources(tuning_tests PRIVATE ${tests})

target_link_libraries(
  tuning_tests 
  PRIVATE project_options
          umpalumpa
          gtest_main
          gmock_main
          cufft
          tbb)

include(GoogleTest)
gtest_discover_tests(tuning_tests)

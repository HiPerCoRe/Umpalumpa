add_executable(correlation_tests)

# list of test files
set(tests correlation_cuda.cpp correlation_cpu.cpp ../../empty.cu # needs to be here so that CMAKE links CUDA libraries
)

if(USE_STARPU)
  list(APPEND tests correlation_starpu.cpp)
endif()

target_sources(correlation_tests PRIVATE ${tests})

target_link_libraries(
  correlation_tests
  PRIVATE project_options
          umpalumpa
          gtest_main
          gmock_main
          cudart
          cufft
          tbb)

include(GoogleTest)
gtest_discover_tests(correlation_tests)

add_executable(fourier_processing_tests)

# list of test files
set(tests fp_cpu.cpp fp_cuda.cpp ../../empty.cu # needs to be here so that CMAKE links CUDA libraries
)

if(USE_STARPU)
  list(APPEND tests fp_starpu.cpp)
endif()

target_sources(fourier_processing_tests PRIVATE ${tests})

target_link_libraries(
  fourier_processing_tests
  PRIVATE project_options
          umpalumpa
          gtest_main
          gmock_main
          cudart
          cufft
          tbb)

include(GoogleTest)
gtest_discover_tests(fourier_processing_tests)

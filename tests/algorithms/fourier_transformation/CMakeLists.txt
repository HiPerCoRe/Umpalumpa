add_executable(fft_tests)

# list of test files
set(tests
    fft_cuda.cpp
    # fft_cpu.cpp
    ../../empty.cu # needs to be here so that CMAKE links CUDA libraries
)

if(USE_STARPU)
  list(APPEND tests fft_starpu.cpp)
endif()

target_sources(fft_tests PRIVATE ${tests})

target_link_libraries(
  fft_tests
  PRIVATE project_options
          umpalumpa
          gtest_main
          gmock_main
          cudart
          cufft
          tbb)

include(GoogleTest)
gtest_discover_tests(fft_tests)

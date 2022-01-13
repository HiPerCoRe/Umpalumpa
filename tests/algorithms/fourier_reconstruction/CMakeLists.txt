add_executable(fourier_reconstruction_tests)

# list of test files
set(tests
    fr_cpu.cpp
    # fr_cuda.cpp
    ../../empty.cu # needs to be here so that CMAKE links CUDA libraries
)

# if(USE_STARPU)
# list(APPEND tests fr_starpu.cpp)
# endif()

target_sources(fourier_reconstruction_tests PRIVATE ${tests})

target_link_libraries(
  fourier_reconstruction_tests
  PRIVATE project_options
          umpalumpa
          gtest_main
          gmock_main
          cudart
          cufft
          tbb)

include(GoogleTest)
gtest_discover_tests(fourier_reconstruction_tests)

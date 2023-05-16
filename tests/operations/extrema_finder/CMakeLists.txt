add_executable(extrema_finder_tests)

# list of test files
set(tests
    extrema_finder_cpu.cpp
    extrema_finder_cuda.cpp
    ../../empty.cu # needs to be here so that CMAKE links CUDA libraries
)

if(USE_STARPU)
  list(APPEND tests extrema_finder_starpu.cpp)
endif()

target_sources(extrema_finder_tests PRIVATE ${tests})

target_link_libraries(
  extrema_finder_tests
  PRIVATE project_options
          umpalumpa
          gtest_main
          gmock_main
          cudart
          cufft
          tbb)

include(GoogleTest)
gtest_discover_tests(extrema_finder_tests)

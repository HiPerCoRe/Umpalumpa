include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG release-1.10.0)
message(STATUS "Fetching Google Tests")
FetchContent_MakeAvailable(googletest)

add_executable(Umpalumpa_Tests)

# list of test files
set(tests
    algorithms/extrema_finder/extrema_finder_cpu.cpp
    algorithms/extrema_finder/extrema_finder_cuda.cpp
    algorithms/fourier_transformation/fft_cuda.cpp
    algorithms/fourier_processing/fp_cpu.cpp
    algorithms/fourier_processing/fp_cuda.cpp
    algorithms/correlation/correlation_cuda.cpp
    empty.cu # needs to be here so that CMAKE links CUDA libraries
)

if(USE_STARPU)
  list(APPEND tests algorithms/extrema_finder/extrema_finder_starpu.cpp)
endif()

target_sources(Umpalumpa_Tests PRIVATE ${tests})

target_link_libraries(
  Umpalumpa_Tests
  PRIVATE project_options
          umpalumpa
          gtest_main
          gmock_main
          cudart
          cufft)

include(GoogleTest)
gtest_discover_tests(Umpalumpa_Tests)

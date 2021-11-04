include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG release-1.10.0)
message(STATUS "Fetching Google Tests")
FetchContent_MakeAvailable(googletest)

add_subdirectory(algorithms/correlation)
add_subdirectory(algorithms/extrema_finder)
add_subdirectory(algorithms/fourier_processing)
add_subdirectory(algorithms/fourier_transformation)
add_subdirectory(data/payload_wrapper)
add_subdirectory(tuning/algorithm_manager)

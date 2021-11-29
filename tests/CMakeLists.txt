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
add_subdirectory(data)
add_subdirectory(tuning/algorithm_manager)
add_subdirectory(tuning/garbage_collection)

file(
  COPY runAll.sh
  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}
  FILE_PERMISSIONS
    OWNER_READ
    OWNER_WRITE
    OWNER_EXECUTE
    GROUP_READ
    GROUP_EXECUTE
    WORLD_READ
    WORLD_EXECUTE)

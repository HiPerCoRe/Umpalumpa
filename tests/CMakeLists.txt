include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG release-1.10.0)
message(STATUS "Fetching Google Tests")
FetchContent_MakeAvailable(googletest)

add_subdirectory(operations/correlation)
add_subdirectory(operations/extrema_finder)
add_subdirectory(operations/fourier_processing)
add_subdirectory(operations/fourier_transformation)
add_subdirectory(operations/fourier_reconstruction)
add_subdirectory(operations/reduction)
add_subdirectory(operations/initialization)
add_subdirectory(data)
add_subdirectory(tuning/strategy_manager)
add_subdirectory(tuning/garbage_collection)
add_subdirectory(tuning/tuning)
add_subdirectory(tuning/gpu_lock)

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

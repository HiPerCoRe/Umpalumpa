add_executable(gpu_lock_tests)

set(tests
    gpu_lock.cpp
    ../../empty.cu
)

target_sources(gpu_lock_tests PRIVATE ${tests})

target_link_libraries(
  gpu_lock_tests 
  PRIVATE project_options
          umpalumpa
          gtest_main
          gmock_main
          cufft
          tbb)

include(GoogleTest)
gtest_discover_tests(gpu_lock_tests)

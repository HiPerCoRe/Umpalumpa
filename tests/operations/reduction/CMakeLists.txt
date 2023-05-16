add_executable(tests)

# list of test files
set(files
    cpu.cpp
    cuda.cpp
    ../../empty.cu # needs to be here so that CMAKE links CUDA libraries
)

if(USE_STARPU)
  list(APPEND files starpu.cpp)
endif()

target_sources(tests PRIVATE ${files})

target_link_libraries(
  tests
  PRIVATE project_options
          umpalumpa
          gtest_main
          gmock_main
          cudart
          cufft
          tbb)

include(GoogleTest)
gtest_discover_tests(tests)

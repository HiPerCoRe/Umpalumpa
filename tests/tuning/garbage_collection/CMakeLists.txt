add_executable(garbage_collection_tests)

set(tests
    garbage_collection.cpp
    ktt_id_tracker.cpp
    ../../empty.cu
)

target_sources(garbage_collection_tests PRIVATE ${tests})

target_link_libraries(
  garbage_collection_tests
  PRIVATE project_options
          umpalumpa
          spdlog
          gtest_main
          gmock_main
          cufft)

include(GoogleTest)
gtest_discover_tests(garbage_collection_tests)

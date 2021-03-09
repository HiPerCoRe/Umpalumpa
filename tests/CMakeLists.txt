
include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        release-1.10.0
)
message("Fetching Google Tests")
FetchContent_MakeAvailable(googletest)


add_executable (Umpalumpa_Tests)
target_sources(Umpalumpa_Tests PRIVATE algorithms/extrema_finder.cpp)
target_link_libraries(
  Umpalumpa_Tests
  PRIVATE project_options umpalumpa gtest_main)
  


include(GoogleTest)
gtest_discover_tests(Umpalumpa_Tests)
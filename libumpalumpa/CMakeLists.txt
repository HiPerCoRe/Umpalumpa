# we don't have to say the type, by default it's static, can be build by setting BUILD_SHARED_LIBS.
add_library(umpalumpa)

if(ENABLE_CLANG_TIDY)
  set_target_properties(umpalumpa PROPERTIES CXX_CLANG_TIDY clang-tidy)
endif()

set_target_properties(umpalumpa PROPERTIES SOVERSION 0 VERSION 0.0.1)

# list of source files
set(umpalumpa_sources algorithms/extrema_finder/single_extrema_finder.cpp)

target_include_directories(umpalumpa PUBLIC ${CMAKE_SOURCE_DIR})

target_sources(umpalumpa PRIVATE ${umpalumpa_sources})
target_link_libraries(umpalumpa PRIVATE project_options project_warnings)

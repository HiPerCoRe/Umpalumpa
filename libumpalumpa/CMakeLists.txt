# we don't have to say the type, by default it's static, can be build by setting BUILD_SHARED_LIBS.
add_library(umpalumpa)

set_target_properties(umpalumpa PROPERTIES SOVERSION 0 VERSION 0.0.1)

# list of headers
set(umpalumpa_headers
algorithms/alg1.hpp)


# list of source files
set(umpalumpa_sources
algorithms/alg1.cpp)

target_include_directories(umpalumpa PUBLIC ${CMAKE_SOURCE_DIR})

target_sources(umpalumpa PRIVATE ${umpalumpa_sources})
target_link_libraries(
  umpalumpa
  PRIVATE project_options
          project_warnings)
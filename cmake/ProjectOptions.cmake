function(set_project_options project_name)
  if(CMAKE_CXX_COMPILER_ID MATCHES ".*Clang")
  option(ENABLE_BUILD_WITH_TIME_TRACE "Enable -ftime-trace to generate time tracing .json files on clang" OFF)
  if(ENABLE_BUILD_WITH_TIME_TRACE)
    target_compile_options(${project_name} INTERFACE -ftime-trace)
  endif()
endif()

target_compile_features(${project_name} INTERFACE cxx_std_14)

endfunction()

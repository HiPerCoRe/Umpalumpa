if(USE_STARPU)
 add_executable(simple_gemm simple_gemm/main.cpp simple_gemm/kernel.cu)
 target_link_libraries(
  simple_gemm
  PRIVATE project_options
          # project_warnings
		  library_ktt
		  library_starpu)
endif()
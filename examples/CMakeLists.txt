if(USE_STARPU)
  add_executable(simple_gemm simple_gemm/main.cpp simple_gemm/kernel.cu)
  target_link_libraries(
    simple_gemm
    PRIVATE project_options
            # project_warnings
            umpalumpa)
endif()

add_subdirectory(flexalign)
add_subdirectory(fourier_reconstruction)

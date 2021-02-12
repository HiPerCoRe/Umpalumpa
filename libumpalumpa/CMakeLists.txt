cmake_minimum_required (VERSION 3.11)

project (libumpalumpa )

add_executable(hello_starpu hello_starpu.c)
add_executable(vector_scale_hybrid vector_scale_hybrid.cpp vector-scale_hybrid_cuda.cu)
add_executable(fftw_benchmark)

target_sources(fftw_benchmark PRIVATE main.cpp empty.cu)

target_link_libraries(fftw_benchmark PRIVATE project_options umpalumpa cufft spdlog)

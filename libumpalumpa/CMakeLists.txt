# we don't have to say the type, by default it's static, can be build by setting BUILD_SHARED_LIBS.
add_library(umpalumpa)

if(ENABLE_CLANG_TIDY)
  set_target_properties(umpalumpa PROPERTIES CXX_CLANG_TIDY clang-tidy)
endif()

set_target_properties(umpalumpa PROPERTIES SOVERSION 0 VERSION 0.0.1)

# list of source files
set(umpalumpa_sources
    utils/system.cpp
    utils/cuda.cpp
    algorithms/extrema_finder/single_extrema_finder_cpu.cpp
    algorithms/extrema_finder/single_extrema_finder_cuda.cpp
    algorithms/fourier_transformation/fft_cuda.cpp
    algorithms/fourier_transformation/fft_cpu.cpp
    algorithms/fourier_processing/fp_cpu.cpp
    algorithms/fourier_processing/fp_cuda.cpp
    algorithms/correlation/correlation_cuda.cpp
    algorithms/correlation/correlation_cpu.cpp
    tuning/ktt_provider.cpp
    tuning/algorithm_manager.cpp
    tuning/ktt_base.cpp)

if(USE_STARPU)
  list(
    APPEND
    umpalumpa_sources
    algorithms/extrema_finder/single_extrema_finder_starpu.cpp
    algorithms/correlation/correlation_starpu.cpp
    algorithms/fourier_transformation/fft_starpu.cpp)
endif()

target_include_directories(umpalumpa PUBLIC ${CMAKE_SOURCE_DIR})

target_sources(umpalumpa PRIVATE ${umpalumpa_sources})
target_link_libraries(umpalumpa PRIVATE project_options project_warnings spdlog)
target_link_libraries(
  umpalumpa
  PUBLIC library_starpu
         library_ktt
         cuda
         library_fftw)

target_compile_definitions(umpalumpa PUBLIC PROJECT_ROOT="${CMAKE_SOURCE_DIR}")

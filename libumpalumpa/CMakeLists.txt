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
    utils/ktt.cpp
    operations/extrema_finder/single_extrema_finder_cpu.cpp
    operations/extrema_finder/single_extrema_finder_cuda.cpp
    operations/fourier_transformation/fft_cuda.cpp
    operations/fourier_transformation/fft_cpu.cpp
    operations/fourier_processing/fp_cpu.cpp
    operations/fourier_processing/fp_cuda.cpp
    operations/fourier_reconstruction/afr.cpp
    operations/fourier_reconstruction/fr_cuda.cpp
    operations/fourier_reconstruction/fr_cpu.cpp
    operations/correlation/correlation_cuda.cpp
    operations/correlation/correlation_cpu.cpp
    operations/reduction/cpu.cpp
    operations/reduction/cuda.cpp
    operations/initialization/cpu.cpp
    operations/initialization/cuda.cpp
    tuning/ktt_provider.cpp
    tuning/strategy_manager.cpp
    tuning/tunable_strategy.cpp
    tuning/ktt_base.cpp)

if(USE_STARPU)
  list(
    APPEND
    umpalumpa_sources
    operations/extrema_finder/single_extrema_finder_starpu.cpp
    operations/correlation/correlation_starpu.cpp
    operations/fourier_processing/fp_starpu.cpp
    operations/fourier_reconstruction/fr_starpu.cpp
    operations/fourier_reconstruction/fr_starpu_kernels.cu
    operations/fourier_transformation/fft_starpu.cpp
    operations/reduction/starpu.cpp
    operations/initialization/starpu.cpp
    utils/starpu.cpp)
endif()

target_include_directories(umpalumpa PUBLIC ${CMAKE_SOURCE_DIR})

target_sources(umpalumpa PRIVATE ${umpalumpa_sources})
target_link_libraries(
  umpalumpa
  PRIVATE project_options
          project_warnings
          spdlog
          library_fftw)
target_link_libraries(umpalumpa PUBLIC library_starpu library_ktt cuda)

target_compile_definitions(umpalumpa PUBLIC PROJECT_ROOT="${CMAKE_SOURCE_DIR}")

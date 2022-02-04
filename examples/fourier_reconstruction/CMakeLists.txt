add_executable(fourier_reconstruction)
set_target_properties(fourier_reconstruction PROPERTIES OUTPUT_NAME "prog")

set(sources
    main.cpp
    fr.cpp
    empty.cu # to link cuda libraries
    fr_cpu.cpp
    fr_cuda.cpp
	)

if(USE_STARPU)
  list(APPEND sources fr_starpu.cpp)
endif()

target_sources(fourier_reconstruction PRIVATE ${sources})

target_link_libraries(fourier_reconstruction PRIVATE project_options umpalumpa cufft spdlog)

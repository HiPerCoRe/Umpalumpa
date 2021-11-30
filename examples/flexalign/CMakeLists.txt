add_executable(flexalign)

set(sources
    main.cpp
    flexalign.cpp
    empty.cu # to link cuda libraries
    flexalign_cpu.cpp
    # flexalign_cuda.cpp
	)

if(USE_STARPU)
  list(APPEND sources flexalign_starpu.cpp)
endif()

target_sources(flexalign PRIVATE ${sources})

target_link_libraries(flexalign PRIVATE project_options umpalumpa cufft spdlog)

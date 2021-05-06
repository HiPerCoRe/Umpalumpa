cmake_minimum_required(VERSION 3.14)

project(umpalumpa LANGUAGES CUDA CXX C)

find_library(CUDART_LIBRARY cudart ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})

# set build types, general compile options etc
include(cmake/StandardProjectSettings.cmake)
# ensure that user is not building in the root
include(cmake/PreventInSourceBuilds.cmake)
# enable cache system (for faster build)
include(cmake/Cache.cmake)
# get all dependencies
include(cmake/UpdateSubmodules.cmake)

# Link this 'library' (phony target) to set the c++ standard / compile-time options requested
add_library(project_options INTERFACE)
include(cmake/ProjectOptions.cmake)
set_project_options(project_options)
# FIXME add sanitizers, doxygen, static analysis, FIXME consider adding precompiled headers support, unity build support

# Link this 'library' to use our warnings
add_library(project_warnings INTERFACE)
include(cmake/CompilerWarnings.cmake)
set_project_warnings(project_warnings)

# get submodules
update_submodules()
add_subdirectory(external)

option(BUILD_SHARED_LIBS "Enable compilation of shared libraries" OFF)

if(CMAKE_CXX_COMPILER_ID MATCHES ".*Clang")
  option(ENABLE_BUILD_WITH_TIME_TRACE "Enable -ftime-trace to generate time tracing .json files on clang" OFF)
  if(ENABLE_BUILD_WITH_TIME_TRACE)
    target_compile_options(project_options INTERFACE -ftime-trace)
  endif()
  option(ENABLE_CLANG_TIDY "Run clang-tidy with the compiler" OFF)
endif()

option(ENABLE_TESTING "Enable Test Builds" ON)
if(ENABLE_TESTING)
  message("Tests enabled")
  enable_testing()
  add_subdirectory(tests)
endif()

option(USE_STARPU "Use StarPU" ON)
add_library(library_starpu INTERFACE)
if(USE_STARPU)
  include(cmake/EnsureStarPU.cmake)
  ensure_starpu(library_starpu)
endif()

add_subdirectory(libumpalumpa)

option(ENABLE_EXAMPLES "Enable Example Builds" ON)
if(ENABLE_EXAMPLES)
  add_subdirectory(examples)
endif()

# FIXME handle installation include(GNUInstallDirs) install(TARGETS libumpalumpa) # since cmake 3.14 install(TARGETS
# libumpalumpa # before cmake 3.14 RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT libumpalumpa_Runtime LIBRARY
# DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT libumpalumpa_Runtime NAMELINK_COMPONENT libumpalumpa_Development # cmake
# 3.12 ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT libumpalumpa_Development )

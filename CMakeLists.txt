cmake_minimum_required(VERSION 3.14)


project (umpalumpa LANGUAGES CUDA CXX C)

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
# FIXME add sanitizers, doxygen, static analysis, 
# FIXME consider adding precompiled headers support, unity build support

# Link this 'library' to use our warnings
add_library(project_warnings INTERFACE)
include(cmake/CompilerWarnings.cmake)
set_project_warnings(project_warnings)

# get submodules
update_submodules()


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


add_subdirectory(libumpalumpa)



# FIXME add STARPU
# find_package (PkgConfig)

# function (ensure_starpu)
	# set (STARPU_REQUESTED_VERSION "starpu-1.3" CACHE STRING "StarPU version to use")

	# check if we have it on the path
	# pkg_check_modules(STARPU QUIET ${STARPU_REQUESTED_VERSION})
	# if (NOT STARPU_FOUND AND NOT DEFINED ENV{STARPU_PATH})
		# message(STATUS "Try to set STARPU_PATH or set PKG_CONFIG_PATH")
	# elseif (DEFINED ENV{STARPU_PATH})
		# but we have STARPU_PATH
		# set (ENV{PKG_CONFIG_PATH} "$ENV{PKG_CONFIG_PATH}:$ENV{STARPU_PATH}/lib/pkgconfig")
	# endif ()
	
	# by now, we either know where STARPU is, or we have no clue
	# pkg_check_modules(STARPU REQUIRED ${STARPU_REQUESTED_VERSION})		
	# if (STARPU_FOUND)
		# include_directories (${STARPU_INCLUDE_DIRS})
		# link_directories       (${STARPU_LIBRARY_DIRS})
		# link_libraries           (${STARPU_LIBRARIES})
	# else ()
		# message(FATAL_ERROR "StarPU not found")
	# endif ()
# endfunction ()

# ensure_starpu ()



# FIXME handle installation
#include(GNUInstallDirs)
#install(TARGETS libumpalumpa) # since cmake 3.14
#install(TARGETS libumpalumpa # before cmake 3.14
#RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
#COMPONENT libumpalumpa_Runtime
#LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
#COMPONENT libumpalumpa_Runtime
#NAMELINK_COMPONENT libumpalumpa_Development # cmake 3.12
#ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
#COMPONENT libumpalumpa_Development
#)

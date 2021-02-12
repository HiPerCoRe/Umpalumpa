cmake_minimum_required (VERSION 3.11)

project (libumpalumpa LANGUAGES CUDA CXX C)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package (PkgConfig)

function (ensure_starpu)
	set (STARPU_REQUESTED_VERSION "starpu-1.3" CACHE STRING "StarPU version to use")

	# check if we have it on the path
	pkg_check_modules(STARPU QUIET ${STARPU_REQUESTED_VERSION})
	if (NOT STARPU_FOUND AND NOT DEFINED ENV{STARPU_PATH})
		message(STATUS "Try to set STARPU_PATH or set PKG_CONFIG_PATH")
	elseif (DEFINED ENV{STARPU_PATH})
		# but we have STARPU_PATH
		set (ENV{PKG_CONFIG_PATH} "$ENV{PKG_CONFIG_PATH}:$ENV{STARPU_PATH}/lib/pkgconfig")
	endif ()
	
	# by now, we either know where STARPU is, or we have no clue
	pkg_check_modules(STARPU REQUIRED ${STARPU_REQUESTED_VERSION})		
	if (STARPU_FOUND)
		include_directories (${STARPU_INCLUDE_DIRS})
		link_directories       (${STARPU_LIBRARY_DIRS})
		link_libraries           (${STARPU_LIBRARIES})
	else ()
		message(FATAL_ERROR "StarPU not found")
	endif ()
endfunction ()

ensure_starpu ()


# define default location of executables
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})


add_subdirectory(libumpalumpa)
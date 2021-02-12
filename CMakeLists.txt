cmake_minimum_required (VERSION 3.1)

project (libumpalumpa)

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

add_executable(hello_starpu libumpalumpa/hello_starpu.c)
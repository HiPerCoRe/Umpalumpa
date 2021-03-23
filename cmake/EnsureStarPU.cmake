find_package (PkgConfig)
function (ensure_starpu library_name)
	set (STARPU_REQUESTED_VERSION "starpu-1.3" CACHE STRING "StarPU version to use")

	# check if we have it on the path
	pkg_check_modules(STARPU QUIET ${STARPU_REQUESTED_VERSION})
	if (NOT STARPU_FOUND AND NOT DEFINED ENV{STARPU_PATH})
		message(STATUS "Try to set STARPU_PATH or set PKG_CONFIG_PATH. You can also:\nsource path_to_starpu/install/bin/estarpu_env")
	elseif (DEFINED ENV{STARPU_PATH})
		# but we have STARPU_PATH
		set (ENV{PKG_CONFIG_PATH} "$ENV{PKG_CONFIG_PATH}:$ENV{STARPU_PATH}/lib/pkgconfig")
	endif ()
	
	# by now, we either know where STARPU is, or we have no clue
	pkg_check_modules(STARPU REQUIRED ${STARPU_REQUESTED_VERSION})		
	if (STARPU_FOUND)
		target_include_directories(${library_name} INTERFACE ${STARPU_INCLUDE_DIRS})
		target_link_directories(${library_name} INTERFACE ${STARPU_LIBRARY_DIRS})
		target_link_libraries(${library_name} INTERFACE ${STARPU_LIBRARIES})
	else ()
		message(FATAL_ERROR "StarPU not found")
	endif ()
endfunction ()
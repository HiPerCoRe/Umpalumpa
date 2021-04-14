
function(update_submodules)

	find_package(Git QUIET)
	if(GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
	# Update submodules as needed
		option(GIT_SUBMODULE "Check submodules during build" ON)
		if(GIT_SUBMODULE)
			message(STATUS "Submodule update")
			# update all of them
			execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive --progress
							WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
							RESULT_VARIABLE GIT_SUBMOD_RESULT)
			if(NOT GIT_SUBMOD_RESULT EQUAL "0")
				message(FATAL_ERROR "git submodule update --init failed with ${GIT_SUBMOD_RESULT}, please checkout submodules")
			endif()
			
			# get newest KTT
			execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive --progress --remote
							WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/external/KTT
							RESULT_VARIABLE GIT_SUBMOD_RESULT)
			if(NOT GIT_SUBMOD_RESULT EQUAL "0")
				message(FATAL_ERROR "git submodule update --init failed with ${GIT_SUBMOD_RESULT}, please checkout submodules")
			endif()
		endif()
	endif()
endfunction()
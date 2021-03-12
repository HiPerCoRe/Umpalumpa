
# make sure that we have premake
set(premake ${CMAKE_CURRENT_SOURCE_DIR}/premake5/premake5)
if(NOT EXISTS ${premake})
	message(STATUS "Downloading premake-5")
	set(premake_version premake-5.0.0-alpha15-linux.tar.gz)
	set(path ${CMAKE_CURRENT_SOURCE_DIR}/premake5)
	file(DOWNLOAD https://github.com/premake/premake-core/releases/download/v5.0.0-alpha15/${premake_version} ${path}/${premake_version})
	execute_process(COMMAND tar -xf ${premake_version} WORKING_DIRECTORY ${path})
endif()

# make sure that KTT is build
set(ktt_dir ${CMAKE_CURRENT_SOURCE_DIR}/KTT)
if(NOT EXISTS ${ktt_dir}/build/)
execute_process(COMMAND ${premake} gmake --no-examples a --no-tutorials WORKING_DIRECTORY ${ktt_dir})
execute_process(COMMAND make WORKING_DIRECTORY ${ktt_dir}/build)
endif()

# create KTT library
add_library(ktt INTERFACE)
target_include_directories(ktt INTERFACE ${ktt_dir}/source/)
target_link_libraries(ktt INTERFACE ${ktt_dir}/build/x86_64_Release/libktt.so)

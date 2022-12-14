find_package(PkgConfig)
function(ensure_fftw library_name)
  # Find FFTW for double precision
  pkg_check_modules(FFTW REQUIRED fftw3)
  if(FFTW_FOUND)
    target_include_directories(${library_name} INTERFACE ${FFTW_INCLUDE_DIRS})
    target_link_directories(${library_name} INTERFACE ${FFTW_LIBRARY_DIRS})
    target_link_libraries(${library_name} INTERFACE ${FFTW_LIBRARIES} fftw3_threads)
  else()
    message(FATAL_ERROR "FFTW (for double precision) not found")
  endif()
  # Find FFTW for single precision
  pkg_check_modules(FFTWF REQUIRED fftw3f)
  if(FFTW_FOUND)
    target_include_directories(${library_name} INTERFACE ${FFTWF_INCLUDE_DIRS})
    target_link_directories(${library_name} INTERFACE ${FFTWF_LIBRARY_DIRS})
    target_link_libraries(${library_name} INTERFACE ${FFTWF_LIBRARIES} fftw3f_threads)
  else()
    message(FATAL_ERROR "FFTW (for single precision) not found")
  endif()
endfunction()

# Collect source files
file(GLOB_RECURSE srcs ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

# Build each source file independently
foreach(source ${srcs})
  get_filename_component(name ${source} NAME_WE)
  add_executable(${name} ${source})
  target_link_libraries(${name} ${LIBS_LINK} ${CUSTOM_LIBS})
  # SET_TARGET_PROPERTIES(${name} PROPERTIES LINK_FLAGS -L${PCL_LIBRARY_DIRS})
endforeach(source)



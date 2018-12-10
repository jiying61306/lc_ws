file(GLOB_RECURSE srcs ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
add_library(CustomLib ${srcs})
target_link_libraries(CustomLib ${LIBS_LINK} )

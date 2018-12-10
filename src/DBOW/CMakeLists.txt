cmake_minimum_required(VERSION 2.8.3)
project(DBOW)
## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  roscpp
  rospy
  pcl_ros
  std_msgs
  tf
  eigen_conversions
  # message_generation
)

find_package(PCL 1.7 REQUIRED)
# include_directories(BEFORE ${PCL_INCLUDE_DIRS})
# link_directories(${PCL_LIBRARY_DIRS})
# add_definitions(${PCL_DEFINITIONS})
if (NOT PCL_FOUND)
    MESSAGE(FATAL_ERROR "PCL not found.\n")
endif (NOT PCL_FOUND)
add_compile_options(-std=c++11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -pedantic")

set(DEPENDENCY_DIR ${CMAKE_CURRENT_BINARY_DIR}/dependencies)
set(DEPENDENCY_INSTALL_DIR ${DEPENDENCY_DIR}/install)

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
MESSAGE("inc " ${OpenCV_INCLUDE_DIRS})
MESSAGE("eigen list: " ${EIGEN3_INCLUDE_DIR})

find_package(Boost REQUIRED)
include_directories(${Boost_INCLUDE_DIR})

# set(LIBS_LINK ${Boost_LIBRARIES} ${OpenCV_LIBS} gflags ${catkin_LIBRARIES}  ${PCL_LIBRARIES})

#add_library(${PROJECT_NAME} SHARED ${SRCS})
#include_directories(include/DBoW2/)
#add_dependencies(${PROJECT_NAME} Dependencies)
#target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${DLib_LIBS})

set(DLib_LIBS /workspace/DBoW2/build/dependencies/install/lib/libDLib.so)
include_directories(/workspace/DBoW2/build/dependencies/install/include)
set(LIBS_LINK ${Boost_LIBRARIES} ${OpenCV_LIBS} ${DLib_LIBS} ${catkin_LIBRARIES} )
MESSAGE("lib list: " ${LIBS_LINK})

catkin_package(
   # CATKIN_DEPENDS message_runtime
#  INCLUDE_DIRS include
#  LIBRARIES pair
#  CATKIN_DEPENDS other_catkin_pkg
#  DEPENDS system_lib
)
# add_message_files(
#   FILES
#   PoseMap.msg
# )
# generate_messages()
include_directories(
  ${catkin_INCLUDE_DIRS}
)
INCLUDE_DIRECTORIES(include)

ADD_SUBDIRECTORY (src)
set(CUSTOM_LIBS ${CUSTOM_LIBS} CustomLib)

ADD_SUBDIRECTORY (tools)

file(COPY images DESTINATION ${CMAKE_BINARY_DIR}/)

#install(TARGETS ${PROJECT_NAME} DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
#install(DIRECTORY include/DBoW2 DESTINATION ${CMAKE_INSTALL_PREFIX}/include)
#install(FILES "${CMAKE_CURRENT_BINARY_DIR}/DBoW2Config.cmake"
#  DESTINATION ${CMAKE_INSTALL_PREFIX}/include/${PROJECT_NAME})
#install(FILES "${PROJECT_BINARY_DIR}/DBoW2Config.cmake" 
#  DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/cmake/DBoW2/)
#install(DIRECTORY ${DEPENDENCY_INSTALL_DIR}/ DESTINATION ${CMAKE_INSTALL_PREFIX} OPTIONAL)


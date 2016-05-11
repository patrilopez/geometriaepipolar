
#Compiler specific options
if(CMAKE_COMPILER_IS_GNUCXX) 
add_definitions(-std=c++11)
endif()
cmake_minimum_required(VERSION 2.8)
project( epi )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( epi epi.cpp )
target_link_libraries( epi ${OpenCV_LIBS} )


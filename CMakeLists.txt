cmake_minimum_required (VERSION 2.6)
project(RememberYou)

option(USE_SSE         "Set to ON to build use SSE"  ON)
option(USE_OPENMP      "Set to ON to build use openmp"  ON)
# The version number.
set(RememberYou_VERSION_MAJOR 1)
set(RememberYou_VERSION_MINOR 0)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")
find_package( OpenCV REQUIRED )

# configure a header file to pass some of the CMake settings
# to the source code
configure_file (
    "${PROJECT_SOURCE_DIR}/RememberYouConfig.h.in"
        "${PROJECT_BINARY_DIR}/RememberYouConfig.h"
)

# add the binary tree to the search path for include files
# so that we will find RememberYouConfig.h
include_directories("${PROJECT_BINARY_DIR}")

#dlib
include_directories ("${PROJECT_SOURCE_DIR}/dlib")
#add_subdirectory (Dlib) 

# add the executable

# 查找当前目录下的所有源文件
# 并将名称保存到 DIR_SRCS 变量
aux_source_directory(. DIR_SRCS)
add_executable(RememberYou ${DIR_SRCS})
target_link_libraries (RememberYou Dlib)

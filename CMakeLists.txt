cmake_minimum_required (VERSION 3.0)
project (Fission-DNN)

# The version number.
set (Fission-DNN_VERSION_MAJOR 2)
set (Fission-DNN_VERSION_MINOR 0)

#    CMake Scripts dir
#set(CMAKE_SCRIPT_DIR ${CMAKE_SOURCE_DIR}/CMakeScripts)

#CMake module path for custom module finding
#set( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SCRIPT_DIR})

find_package (CUDA REQUIRED)
include_directories (${CUDA_INCLUDE_DIRS})
#include_directories (${CUDA_SDK_ROOT_DIR})
MESSAGE("${CUDA_INCLUDE_DIRS}/../samples/common/inc")
include_directories ("${CUDA_INCLUDE_DIRS}/../samples/common/inc")
#/usr/local/cuda/NVIDIA_CUDA-7.0_Samples/common/inc")
set(CUDA_SEPARABLE_COMPILATION ON)

# opencv
find_package( OpenCV REQUIRED )
INCLUDE_DIRECTORIES( ${OPENCV_INCLUDE_DIR} )
set(OpenCV_LIBRARIES ${OpenCV_LIBS})

#Protobuf
#find_package(Protobuf REQUIRED)
#INCLUDE_DIRECTORIES( ${PROTOBUF_INCLUDE_DIR} )
#set(PROTOBUF_LIBRARIES ${PROTOBUF_LIBS})
#PROTOBUF_GENERATE_CPP(PROTO_SRCS PROTO_HDRS proto/datum.proto)
# 生成一个静态库protolib,在链接库文件件需要 指定
#add_library(protolib ${PROTO_SRCS} ${PROTO_HDRS}

find_package( Protobuf REQUIRED  )
include_directories(SYSTEM ${PROTOBUF_INCLUDE_DIR})
list(APPEND Caffe_LINKER_LIBS ${PROTOBUF_LIBRARIES})
if(EXISTS ${PROTOBUF_PROTOC_EXECUTABLE})
      message(STATUS "Found PROTOBUF Compiler: ${PROTOBUF_PROTOC_EXECUTABLE}")
  else()
        message(FATAL_ERROR "Could not find PROTOBUF Compiler")
endif()
set(proto_gen_folder "${PROJECT_BINARY_DIR}/proto")
#include_directories("${PROJECT_BINARY_DIR}/include")
set(output_dir "${PROJECT_BINARY_DIR}/proto")
PROTOBUF_GENERATE_CPP(PROTO_SRCS PROTO_HDRS proto/datum.proto)
set(PROTOBUF_GENERATE_CPP_APPEND_PATH TRUE)

#LMDB
find_path(LMDB_INCLUDE_DIR NAMES  lmdb.h PATHS "$ENV{LMDB_DIR}/include")
find_library(LMDB_LIBRARIES NAMES lmdb   PATHS "$ENV{LMDB_DIR}/lib" )

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(LMDB DEFAULT_MSG LMDB_INCLUDE_DIR LMDB_LIBRARIES)


SET(CUDA_CUDNN_LIBRARY "/usr/local/cuda/lib64/libcudnn.so")
SET(GOOGLE_GLOG_LIBRARY "/usr/lib/x86_64-linux-gnu/libglog.so")
SET(ATLAS_LIBRARY "/usr/lib/atlas-base/libcblas.so.3.0")

include_directories(${PROJECT_SOURCE_DIR})

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -g -rdynamic")

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
  -gencode arch=compute_20,code=sm_20
  -gencode arch=compute_20,code=sm_21
  -gencode arch=compute_30,code=sm_30
  -gencode arch=compute_35,code=sm_35
  -gencode arch=compute_50,code=sm_50
  -gencode arch=compute_50,code=compute_50
  -rdc=true
  )

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
  )

MESSAGE(${CUDA_NVCC_FLAGS})

#add_library(${EXAMPLE_NAME}.o OBJECT ${source})
#set(EXAMPLE_OBJ_LIB $<TARGET_OBJECTS:${EXAMPLE_NAME}.o>)


file(GLOB Fission-DNN_CU_SOURCES  RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} 
    "common/*"
    "main.cpp"
    "config/*"
    "common/util/*"
    "examples/mnist/*"
    "build/datum.pb.*"
    "layers/*"
    )
    #  EXCLUDE_FROM_ALL "build/*")

    cuda_add_executable(Fission-DNN ${Fission-DNN_CU_SOURCES} ${PROTO_SRCS} ${PROTO_HDRS})
    #cuda_add_library(Fission-DNN STATIC ${Fission-DNN_CPP_SOURCES})
target_link_libraries(Fission-DNN
  ${OpenCV_LIBRARIES}
  ${CUDA_CUBLAS_LIBRARIES}
  ${CUDA_curand_LIBRARY}
  ${CUDA_LIBRARIES}
  ${CUDA_CUDNN_LIBRARY}
  ${GOOGLE_GLOG_LIBRARY}
  ${ATLAS_LIBRARY}
  ${LMDB_LIBRARIES}
  protobuf
  )
CUDA_ADD_CUBLAS_TO_TARGET(Fission-DNN)

#add_subdirectory(net)

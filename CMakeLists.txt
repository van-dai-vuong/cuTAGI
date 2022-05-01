# CMAKE VERSION
# NOTE*: We only test this file using CUDA toolkit 10.2
cmake_minimum_required(VERSION 3.16)

############################################################################################
## PPROJECT
############################################################################################

PROJECT(
  cuTAGI
  VERSION 0.1.0
  DESCRIPTION "C++/CUDA library for Tractable Approximate Gaussian Inference"
  LANGUAGES CXX CUDA
)

set(CUTAGI_VERSION "${CMAKE_PROJECT_VERSION}")

############################################################################################
## C++ COMPILER SETUP
############################################################################################
# Configuration 
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
	message(STATUS "Build type is set to 'Release'.")
	set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
	set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "RelWithDebInfo")
endif()

# CUDA Path 
if (MSVC)
else()
	set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)
endif()
find_library(
	${CUDA_TOOLKIT_ROOT_DIR}/lib64
	${CUDA_TOOLKIT_ROOT_DIR}/lib
)

if (APPLE)
	set(CMAKE_MACOSX_RPATH ON)
endif()

if (MSVC)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /D_CRT_SECURE_NO_WARNINGS")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP24")
else()
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
endif()

set(CMAKE_CXX_STANDARD 14)    # C++ version
set(CMAKE_CXX_EXTENSIONS OFF) # Disable GNU extenstions

############################################################################################
## CUDA COMPILER SETUP
############################################################################################
set(CMAKE_CUDA_STANDARD 14) 
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CUDA_LINK_LIBRARIES_KEYWORD PUBLIC)

# Set compiler options
if (MSVC)
	list(APPEND CUDA_NVCC_FLAGS "-Xcompiler=-bigobj")
else()
	list(APPEND CUDA_NVCC_FLAGS "-Xcompiler=-mf16c")
	list(APPEND CUDA_NVCC_FLAGS "-Xcompiler=-Wno-float-conversion")
	list(APPEND CUDA_NVCC_FLAGS "-Xcompiler=-fno-strict-aliasing")
	list(APPEND CUDA_NVCC_FLAGS "-Xcompiler=-fPIC")
	set(CUDA_TOOLKIT_ROOT_DIR /opt/cuda/targets/x86_64-linux)
endif()

# NOTE: This might need to change for higher CUDA version 
list(APPEND CUDA_NVCC_FLAGS "--expt-extended-lambda")
list(APPEND CUDA_NVCC_FLAGS "--expt-relaxed-constexpr")

# Header files for source code
include_directories("include")

# Sources code *.cu *.cpp files
set(SOURCES
  src/common.cpp
  src/cost.cpp 
  src/dataloader.cpp 
  src/data_transfer.cu 
  src/feed_forward.cu 
  src/global_param_update.cu 
  src/indices.cpp 
  src/net_init.cpp 
  src/net_prop.cpp 
  src/param_feed_backward.cu 
  src/state_feed_backward.cu 
  src/task.cu 
  src/user_input.cpp 
  src/utils.cpp
)

# Output binary folder der for different mode
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR})

add_library(cutagi STATIC ${SOURCES})
set_target_properties(cutagi PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_target_properties(cutagi PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Set CUDA flags only on target i.e. only for files that are compiled for CUDA 
target_compile_options(cutagi PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_NVCC_FLAGS}>)

# Executable code i.e. application
add_executable(main main.cpp)
target_link_libraries(main PUBLIC cutagi)
#target_compile_options(main PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_NVCC_FLAGS}>)


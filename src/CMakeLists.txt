cmake_minimum_required(VERSION 3.9)
################################
# cmake policy selections
################################
# avoid default mac osx rpath settings for cmake 3.0
cmake_policy(SET CMP0042 OLD)
# avoid warnings for project commands w/o VERSION
cmake_policy(SET CMP0048 NEW)


################################
# Ascent
################################

project(ultroids VERSION "0.0.1")

################################
# Build Options
################################
option(BUILD_SHARED_LIBS  "Build shared libraries"    ON)

option(ENABLE_MPI         "Build MPI Support"         ON)
option(ENABLE_CUDA        "Build CUDA Support"        OFF)
option(ENABLE_OPENMP      "Build OpenMP Support"      OFF)

set(ENABLE_TESTS OFF CACHE Bool "")
set(ENABLE_EXAMPLES OFF CACHE Bool "")
set(ENABLE_DOCUMENTATION OFF CACHE Bool "")
################################
# Set BLT Options
################################
# don't use BLT's all warnings feature
set(ENABLE_ALL_WARNINGS OFF CACHE BOOL "")

################################
# Init BLT
################################
# if BLT_SOURCE_DIR is not set - use "blt" as default
if(NOT BLT_SOURCE_DIR)
    set(BLT_SOURCE_DIR "blt")
endif()

if (NOT BLT_CXX_STD)
  set(BLT_CXX_STD "c++11" CACHE STRING "")
endif()


# init blt using BLT_SOURCE_DIR
include(${BLT_SOURCE_DIR}/SetupBLT.cmake)

add_subdirectory(CHAI)

################################
# Basic CMake Setup
################################
include(cmake/CMakeBasics.cmake)

add_subdirectory(ultroids)

cmake_minimum_required(VERSION 3.24)
project(q9 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build header-only library.
add_library(q9 INTERFACE)
target_include_directories(q9 INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
  $<INSTALL_INTERFACE:include/>  # <prefix>/include
  )

add_subdirectory(q9-eval-mapping)

cmake_minimum_required(VERSION 3.24)
project(q9 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Enable MLIR support in MQT Core
set(BUILD_MQT_CORE_MLIR ON CACHE BOOL "Enable MLIR support in MQT Core" FORCE)

include(FetchContent)

FetchContent_Declare(
  mqt-core
  GIT_REPOSITORY https://github.com/munich-quantum-toolkit/core.git
  GIT_TAG v3.4.1
  FIND_PACKAGE_ARGS 3.4.1)

FetchContent_MakeAvailable(mqt-core)

add_executable(q9 main.cpp)
target_link_libraries(q9 PRIVATE MQT::CoreIR)

# Enable MLIR support in MQT Core
set(BUILD_MQT_CORE_MLIR ON CACHE BOOL "Enable MLIR support in MQT Core" FORCE)

include(FetchContent)

FetchContent_Declare(
  mqt-core
  GIT_REPOSITORY https://github.com/munich-quantum-toolkit/core.git
  GIT_TAG main)

FetchContent_MakeAvailable(mqt-core)

find_package(MLIR REQUIRED CONFIG)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/mlir/include)
include_directories(${PROJECT_BINARY_DIR}/mlir/include)

add_executable(q9-eval-mapping q9-eval-mapping.cpp)
target_link_libraries(q9-eval-mapping PRIVATE q9 MQT::CoreQASM
          MQT::CoreIR MLIRQCTranslation MLIRParser
          MLIRQCProgramBuilder
          MLIRQCOUtils
          MLIRQCToQCO
          MLIRQCOToQC
          MLIRQCOTransforms)
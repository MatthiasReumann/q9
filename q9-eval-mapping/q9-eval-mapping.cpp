#include "mlir/Conversion/QCToQCO/QCToQCO.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/IR/Operation.h"
#include "sc/configuration/Configuration.hpp"
#include "sc/configuration/Heuristic.hpp"
#include "sc/configuration/InitialLayout.hpp"
#include "sc/configuration/Layering.hpp"
#include "sc/configuration/LookaheadHeuristic.hpp"
#include "sc/heuristic/HeuristicMapper.hpp"
#include <chrono>
#include <filesystem>
#include <iostream>
#include <ir/QuantumComputation.hpp>
#include <memory>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/QC/Translation/TranslateQuantumComputationToQC.h>
#include <mlir/Dialect/QCO/IR/QCODialect.h>
#include <mlir/Dialect/QCO/Transforms/Passes.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Value.h>
#include <mlir/Pass/PassInstrumentation.h>
#include <mlir/Pass/PassManager.h>
#include <q9/q9.h>
#include <qasm3/Importer.hpp>
#include <sc/Architecture.hpp>
#include <stdexcept>
#include <vector>

namespace {

class MQT : q9::Task {
  struct PassExecutionStats : public mlir::PassInstrumentation {
    PassExecutionStats(q9::Statistics &stats) : stats_(&stats) {}

    void runBeforePass([[maybe_unused]] mlir::Pass *pass,
                       mlir::Operation *op) override {
      nswapsPrev_ = countSwaps(op);
      t0_ = std::chrono::steady_clock::now();
    }

    void runAfterPass(mlir::Pass *pass, mlir::Operation *op) override {
      const auto t1 = std::chrono::steady_clock::now();
      const auto duration =
          std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0_)
              .count();
      const auto nswapsAfter = countSwaps(op);

      stats_->add("time(ms)", duration);
      stats_->add("nswaps", nswapsAfter - nswapsPrev_);
    }

  private:
    std::size_t countSwaps(mlir::Operation *op) {
      std::size_t cnt{0};
      op->walk([&cnt](mlir::qco::SWAPOp) { ++cnt; });
      return cnt;
    }

    q9::Statistics *stats_;
    std::chrono::steady_clock::time_point t0_;
    std::size_t nswapsPrev_;
  };

public:
  MQT(const std::string &name) : q9::Task(name) {
    mlir::DialectRegistry registry;
    registry.insert<mlir::qco::QCODialect, mlir::arith::ArithDialect,
                    mlir::func::FuncDialect>();

    context_ = std::make_unique<mlir::MLIRContext>(registry);
    context_->loadAllAvailableDialects();
  }

  q9::Statistics run(q9::ProgramRepresentation &repr) override {
    auto pm = mlir::PassManager(context_.get());
    auto module = q9::Adapter::toQCO(repr, context_.get());

    q9::Statistics stats;
    pm.addPass(mlir::qco::createMappingPass(
        mlir::qco::MappingPassOptions{.nlookahead = 15,
                                      .alpha = 1,
                                      .lambda = 0.5,
                                      .niterations = 1,
                                      .ntrials = 8,
                                      .seed = 333}));
    pm.addInstrumentation(std::make_unique<PassExecutionStats>(stats));
    pm.addPass(mlir::createQCToQCO());
    if (pm.run(module.get()).failed()) {
      throw std::runtime_error("MQT compiler collection mapping pass failed.");
    }

    return stats;
  }

private:
  std::unique_ptr<mlir::MLIRContext> context_;
};

class QMAP : q9::Task {
public:
  using q9::Task::Task;
  q9::Statistics run(q9::ProgramRepresentation &repr) override {
    Architecture architecture(
        120,
        {{0, 12},    {12, 0},    {0, 1},     {1, 0},     {1, 13},    {13, 1},
         {1, 2},     {2, 1},     {2, 14},    {14, 2},    {2, 3},     {3, 2},
         {3, 15},    {15, 3},    {3, 4},     {4, 3},     {4, 16},    {16, 4},
         {4, 5},     {5, 4},     {5, 17},    {17, 5},    {5, 6},     {6, 5},
         {6, 18},    {18, 6},    {6, 7},     {7, 6},     {7, 19},    {19, 7},
         {7, 8},     {8, 7},     {8, 20},    {20, 8},    {8, 9},     {9, 8},
         {9, 21},    {21, 9},    {9, 10},    {10, 9},    {10, 22},   {22, 10},
         {10, 11},   {11, 10},   {11, 23},   {23, 11},   {12, 24},   {24, 12},
         {12, 13},   {13, 12},   {13, 25},   {25, 13},   {13, 14},   {14, 13},
         {14, 26},   {26, 14},   {14, 15},   {15, 14},   {15, 27},   {27, 15},
         {15, 16},   {16, 15},   {16, 28},   {28, 16},   {16, 17},   {17, 16},
         {17, 29},   {29, 17},   {17, 18},   {18, 17},   {18, 30},   {30, 18},
         {18, 19},   {19, 18},   {19, 31},   {31, 19},   {19, 20},   {20, 19},
         {20, 32},   {32, 20},   {20, 21},   {21, 20},   {21, 33},   {33, 21},
         {21, 22},   {22, 21},   {22, 34},   {34, 22},   {22, 23},   {23, 22},
         {23, 35},   {35, 23},   {24, 36},   {36, 24},   {24, 25},   {25, 24},
         {25, 37},   {37, 25},   {25, 26},   {26, 25},   {26, 38},   {38, 26},
         {26, 27},   {27, 26},   {27, 39},   {39, 27},   {27, 28},   {28, 27},
         {28, 40},   {40, 28},   {28, 29},   {29, 28},   {29, 41},   {41, 29},
         {29, 30},   {30, 29},   {30, 42},   {42, 30},   {30, 31},   {31, 30},
         {31, 43},   {43, 31},   {31, 32},   {32, 31},   {32, 44},   {44, 32},
         {32, 33},   {33, 32},   {33, 45},   {45, 33},   {33, 34},   {34, 33},
         {34, 46},   {46, 34},   {34, 35},   {35, 34},   {35, 47},   {47, 35},
         {36, 48},   {48, 36},   {36, 37},   {37, 36},   {37, 49},   {49, 37},
         {37, 38},   {38, 37},   {38, 50},   {50, 38},   {38, 39},   {39, 38},
         {39, 51},   {51, 39},   {39, 40},   {40, 39},   {40, 52},   {52, 40},
         {40, 41},   {41, 40},   {41, 53},   {53, 41},   {41, 42},   {42, 41},
         {42, 54},   {54, 42},   {42, 43},   {43, 42},   {43, 55},   {55, 43},
         {43, 44},   {44, 43},   {44, 56},   {56, 44},   {44, 45},   {45, 44},
         {45, 57},   {57, 45},   {45, 46},   {46, 45},   {46, 58},   {58, 46},
         {46, 47},   {47, 46},   {47, 59},   {59, 47},   {48, 60},   {60, 48},
         {48, 49},   {49, 48},   {49, 61},   {61, 49},   {49, 50},   {50, 49},
         {50, 62},   {62, 50},   {50, 51},   {51, 50},   {51, 63},   {63, 51},
         {51, 52},   {52, 51},   {52, 64},   {64, 52},   {52, 53},   {53, 52},
         {53, 65},   {65, 53},   {53, 54},   {54, 53},   {54, 66},   {66, 54},
         {54, 55},   {55, 54},   {55, 67},   {67, 55},   {55, 56},   {56, 55},
         {56, 68},   {68, 56},   {56, 57},   {57, 56},   {57, 69},   {69, 57},
         {57, 58},   {58, 57},   {58, 70},   {70, 58},   {58, 59},   {59, 58},
         {59, 71},   {71, 59},   {60, 72},   {72, 60},   {60, 61},   {61, 60},
         {61, 73},   {73, 61},   {61, 62},   {62, 61},   {62, 74},   {74, 62},
         {62, 63},   {63, 62},   {63, 75},   {75, 63},   {63, 64},   {64, 63},
         {64, 76},   {76, 64},   {64, 65},   {65, 64},   {65, 77},   {77, 65},
         {65, 66},   {66, 65},   {66, 78},   {78, 66},   {66, 67},   {67, 66},
         {67, 79},   {79, 67},   {67, 68},   {68, 67},   {68, 80},   {80, 68},
         {68, 69},   {69, 68},   {69, 81},   {81, 69},   {69, 70},   {70, 69},
         {70, 82},   {82, 70},   {70, 71},   {71, 70},   {71, 83},   {83, 71},
         {72, 84},   {84, 72},   {72, 73},   {73, 72},   {73, 85},   {85, 73},
         {73, 74},   {74, 73},   {74, 86},   {86, 74},   {74, 75},   {75, 74},
         {75, 87},   {87, 75},   {75, 76},   {76, 75},   {76, 88},   {88, 76},
         {76, 77},   {77, 76},   {77, 89},   {89, 77},   {77, 78},   {78, 77},
         {78, 90},   {90, 78},   {78, 79},   {79, 78},   {79, 91},   {91, 79},
         {79, 80},   {80, 79},   {80, 92},   {92, 80},   {80, 81},   {81, 80},
         {81, 93},   {93, 81},   {81, 82},   {82, 81},   {82, 94},   {94, 82},
         {82, 83},   {83, 82},   {83, 95},   {95, 83},   {84, 96},   {96, 84},
         {84, 85},   {85, 84},   {85, 97},   {97, 85},   {85, 86},   {86, 85},
         {86, 98},   {98, 86},   {86, 87},   {87, 86},   {87, 99},   {99, 87},
         {87, 88},   {88, 87},   {88, 100},  {100, 88},  {88, 89},   {89, 88},
         {89, 101},  {101, 89},  {89, 90},   {90, 89},   {90, 102},  {102, 90},
         {90, 91},   {91, 90},   {91, 103},  {103, 91},  {91, 92},   {92, 91},
         {92, 104},  {104, 92},  {92, 93},   {93, 92},   {93, 105},  {105, 93},
         {93, 94},   {94, 93},   {94, 106},  {106, 94},  {94, 95},   {95, 94},
         {95, 107},  {107, 95},  {96, 108},  {108, 96},  {96, 97},   {97, 96},
         {97, 109},  {109, 97},  {97, 98},   {98, 97},   {98, 110},  {110, 98},
         {98, 99},   {99, 98},   {99, 111},  {111, 99},  {99, 100},  {100, 99},
         {100, 112}, {112, 100}, {100, 101}, {101, 100}, {101, 113}, {113, 101},
         {101, 102}, {102, 101}, {102, 114}, {114, 102}, {102, 103}, {103, 102},
         {103, 115}, {115, 103}, {103, 104}, {104, 103}, {104, 116}, {116, 104},
         {104, 105}, {105, 104}, {105, 117}, {117, 105}, {105, 106}, {106, 105},
         {106, 118}, {118, 106}, {106, 107}, {107, 106}, {107, 119}, {119, 107},
         {108, 109}, {109, 108}, {109, 110}, {110, 109}, {110, 111}, {111, 110},
         {111, 112}, {112, 111}, {112, 113}, {113, 112}, {113, 114}, {114, 113},
         {114, 115}, {115, 114}, {115, 116}, {116, 115}, {116, 117}, {117, 116},
         {117, 118}, {118, 117}, {118, 119}, {119, 118}});

    HeuristicMapper mapper(q9::Adapter::toQuantumComputation(repr),
                           architecture);

    Configuration settings{};
    // settings.heuristic = Heuristic::GateCountSumDistance;
    // settings.layering = Layering::DisjointQubits;
    // settings.initialLayout = InitialLayout::Identity;
    // settings.preMappingOptimizations = false;
    // settings.postMappingOptimizations = false;
    // settings.lookaheadHeuristic = LookaheadHeuristic::GateCountSumDistance;
    // settings.nrLookaheads = 15;
    // settings.lookaheadFactor = 0.5;
    mapper.map(settings);
    auto &result = mapper.getResults();

    // Run mapping and collect stats.
    q9::Statistics stats;
    stats.add("time(ms)", std::chrono::duration_cast<std::chrono::milliseconds>(
                              std::chrono::duration<double>(result.time))
                              .count());
    stats.add("nswaps", result.output.swaps);
    return stats;
  }
};
} // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    return 1;
  }

  const std::filesystem::path inputPath(argv[1]);
  std::vector<std::filesystem::path> inputFiles;

  if (std::filesystem::is_regular_file(inputPath)) {
    inputFiles.push_back(inputPath);
  } else if (std::filesystem::is_directory(inputPath)) {
    for (const auto &entry : std::filesystem::directory_iterator(inputPath)) {
      if (entry.is_regular_file() && entry.path().extension() == ".qasm") {
        inputFiles.push_back(entry.path());
      }
    }
  } else {
    throw std::invalid_argument("Invalid input path");
  }

  std::cout << "filename;qmap_time;qmap_nswaps;mqt_time;mqt_nswaps\n";
  for (const auto &inputFile : inputFiles) {
    auto repr = q9::ProgramRepresentation(inputFile.string());

    auto qmapRes = QMAP("QMAP: Mapping").run(repr);
    auto mqtRes = MQT("MQT Compiler Collection: Mapping").run(repr);

    std::cout << inputFile.filename().c_str() << ';' << qmapRes.get("time(ms)") << ';'
              << qmapRes.get("nswaps") << ';' << mqtRes.get("time(ms)") << ';'
              << mqtRes.get("nswaps") << '\n';
  }

  return 0;
}

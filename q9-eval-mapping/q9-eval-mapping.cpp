#include "mlir/Conversion/QCToQCO/QCToQCO.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include <chrono>
#include <iostream>
#include <ir/QuantumComputation.hpp>
#include <memory>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/QC/Translation/TranslateQuantumComputationToQC.h>
#include <mlir/Dialect/QCO/IR/QCODialect.h>
#include <mlir/Dialect/QCO/Transforms/Passes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/Value.h>
#include <mlir/Pass/PassInstrumentation.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Pass/PassRegistry.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/WalkResult.h>
#include <q9/q9.h>
#include <qasm3/Importer.hpp>

namespace {

struct MappingStats {
  std::size_t numSwaps{0};
  std::size_t time{0};
};

class MqtMappingTask : public q9::Task<MappingStats> {
  struct StatsInstrumentation : public mlir::PassInstrumentation {
    StatsInstrumentation(MappingStats &stats) : stats(&stats) {}

    void runBeforePass([[maybe_unused]] mlir::Pass *pass,
                       mlir::Operation *op) override {
      op->walk([this](mlir::qco::SWAPOp) { ++swapsBefore; });
      t0 = std::chrono::steady_clock::now();
    }

    void runAfterPass(mlir::Pass *pass, mlir::Operation *op) override {
      const auto t1 = std::chrono::steady_clock::now();
      stats->time =
          std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0)
              .count();
      op->walk([this](mlir::qco::SWAPOp) { ++stats->numSwaps; });
      stats->numSwaps -= swapsBefore;
    }

  private:
    MappingStats *stats;
    std::chrono::steady_clock::time_point t0;
    std::size_t swapsBefore;
  };

public:
  void prepare() override {
    mlir::DialectRegistry registry;
    registry.insert<mlir::qco::QCODialect, mlir::arith::ArithDialect,
                    mlir::func::FuncDialect>();
    context = std::make_unique<mlir::MLIRContext>();
    context = std::make_unique<mlir::MLIRContext>();
    context->appendDialectRegistry(registry);
    context->loadAllAvailableDialects();

    // Import MQT-IR from QASM representation.
    qc::QuantumComputation qc =
        qasm3::Importer::imports(std::string(getProgramRepresentation()));
    // Translate from MQT-IR to QC dialect.
    module = mlir::translateQuantumComputationToQC(context.get(), qc);
    // Conversion to QCO dialect.
    auto pm = mlir::PassManager(context.get());
    pm.addPass(mlir::createQCToQCO());
    std::ignore = pm.run(*module);
  }

  MappingStats run() override {
    auto pm = mlir::PassManager(context.get());
    pm.addPass(mlir::qco::createMappingPass(
        mlir::qco::MappingPassOptions{.nlookahead = 15,
                                      .alpha = 1,
                                      .lambda = 0.5,
                                      .niterations = 2,
                                      .ntrials = 4,
                                      .seed = 42}));
    pm.addInstrumentation(std::make_unique<StatsInstrumentation>(stats));
    auto res = pm.run(module.get());
    return stats;
  }

  void cleanup() override {}

private:
  MappingStats stats;
  std::unique_ptr<mlir::MLIRContext> context;
  mlir::OwningOpRef<mlir::ModuleOp> module;
};
} // namespace

int main(int argc, char **argv) {
  MqtMappingTask mqtMapping;
  mqtMapping.importf(argv[1]);
  mqtMapping.prepare();
  auto stats = mqtMapping.run();
  mqtMapping.cleanup();
  std::cout << "time in ms: " << stats.time << '\n';
  std::cout << "number of swaps: " << stats.numSwaps << '\n';

  return 0;
}

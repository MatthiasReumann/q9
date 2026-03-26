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
                                      .niterations = 4,
                                      .ntrials = 4,
                                      .seed = 42}));
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
        9, {{0, 3}, {3, 0}, {0, 1}, {1, 0}, {1, 4}, {4, 1}, {1, 2}, {2, 1},
            {2, 5}, {5, 2}, {3, 6}, {6, 3}, {3, 4}, {4, 3}, {4, 7}, {7, 4},
            {4, 5}, {5, 4}, {5, 8}, {8, 5}, {6, 7}, {7, 6}, {7, 8}, {8, 7}});

    HeuristicMapper mapper(q9::Adapter::toQuantumComputation(repr),
                           architecture);

    Configuration settings{};
    settings.heuristic = Heuristic::GateCountSumDistance;
    settings.layering = Layering::DisjointQubits;
    settings.initialLayout = InitialLayout::Dynamic;
    settings.preMappingOptimizations = false;
    settings.postMappingOptimizations = false;
    settings.lookaheadHeuristic = LookaheadHeuristic::GateCountSumDistance;
    settings.nrLookaheads = 15;
    settings.lookaheadFactor = 0.5;
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

  auto repr = q9::ProgramRepresentation(argv[1]);
  auto qmapRes = QMAP("QMAP: Mapping").run(repr);
  auto mqtRes = MQT("MQT Compiler Collection: Mapping").run(repr);

  qmapRes.print();
  mqtRes.print();

  return 0;
}

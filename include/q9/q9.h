#pragma once

#include "mlir/Conversion/QCToQCO/QCToQCO.h"
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <ir/QuantumComputation.hpp>
#include <mlir/Dialect/QC/Translation/TranslateQuantumComputationToQC.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Pass/PassManager.h>
#include <qasm3/Importer.hpp>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
namespace q9 {
class Statistics {
public:
  void add(const std::string &name, const std::size_t value) {
    stats_.emplace(name, value);
  }

  void print() {
    if (stats_.empty()) {
      std::cout << "Statistics: (none)\n";
      return;
    }

    std::vector<std::pair<std::string, std::size_t>> entries(stats_.begin(),
                                                             stats_.end());
    std::sort(
        entries.begin(), entries.end(),
        [](const auto &lhs, const auto &rhs) { return lhs.first < rhs.first; });

    std::size_t maxNameLength = 0;
    for (const auto &[name, _] : entries) {
      maxNameLength = std::max(maxNameLength, name.size());
    }

    std::cout << "Statistics:\n";
    for (const auto &[name, value] : entries) {
      std::cout << "  - " << std::left
                << std::setw(static_cast<int>(maxNameLength)) << name << " : "
                << value << '\n';
    }
  }

private:
  std::unordered_map<std::string, std::size_t> stats_;
};

// TODO: JEFF
class ProgramRepresentation {
public:
  ProgramRepresentation(const std::string &filename)
      : progr_(qasm3::Importer::importf(filename)) {}

  qc::QuantumComputation get() const { return progr_; }

private:
  qc::QuantumComputation progr_;
};

class Adapter {
public:
  static mlir::OwningOpRef<mlir::ModuleOp>
  toQC(const ProgramRepresentation &repr, mlir::MLIRContext *context) {
    return mlir::translateQuantumComputationToQC(context, repr.get());
  }

  static mlir::OwningOpRef<mlir::ModuleOp>
  toQCO(const ProgramRepresentation &repr, mlir::MLIRContext *context) {
    auto pm = mlir::PassManager(context);
    auto module = toQC(repr, context);
    pm.addPass(mlir::createQCToQCO());
    if (pm.run(module.get()).failed()) {
      throw std::runtime_error("Conversion from QC to QCO dialect failed!");
    }
    return module;
  }

  static qc::QuantumComputation
  toQuantumComputation(const ProgramRepresentation &repr) {
    return repr.get();
  }
};

class Task {
public:
  Task(const std::string &name) : name_(name) {}
  
  virtual Statistics run(ProgramRepresentation &repr) = 0;
  virtual ~Task() = default;

private:
  std::string name_;
};
}; // namespace q9
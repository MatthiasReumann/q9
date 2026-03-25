#pragma once

#include <cstddef>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
namespace q9 {
template <class Result> class Task {
public:
  void importf(std::string path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
      throw std::runtime_error("Failed to open qasm file: " + path);
    }
    programRepr_.assign(std::istreambuf_iterator<char>(in),
                        std::istreambuf_iterator<char>());
  };

  virtual void prepare() = 0;
  virtual Result run() = 0;
  virtual void cleanup() = 0;

protected:
  std::string_view getProgramRepresentation() const { return programRepr_; }

private:
  std::string programRepr_;
};
template <typename Input> class Stage {
public:
  virtual void run(Input &) = 0;
  virtual ~Stage() = default;
};
}; // namespace q9
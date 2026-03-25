#include <iostream>

#include "mqt-core/ir/QuantumComputation.hpp"

int main() {
    std::cout << "Hello, World!" << std::endl;

    mqt::qc::QuantumComputation qc(2U);
    qc.h(0U);
    qc.cx(0U, 1U);
    std::cout << qc << std::endl;

    return 0;
}

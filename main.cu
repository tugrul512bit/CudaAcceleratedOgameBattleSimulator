#include "Simulator.cuh"

int main() {
    bool debug = true;
    int randomSeed = rand();
    Simulator simulator(debug, randomSeed);
    simulator.demo();
    return 0;
}

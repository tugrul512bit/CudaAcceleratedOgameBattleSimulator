#include "Simulator.cuh"

int main() {
    bool debug = true;
    int randomSeed = rand();
    Simulator simulator(debug, randomSeed);
    // Adding 1000 ships of type 0.
    simulator.addShips(1000, 0);
	return 0;
}

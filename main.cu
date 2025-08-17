#include "Simulator.cuh"

int main() {
    bool debug = true;
    int randomSeed = rand();
    int randomSeed2 = rand();
    int lightFighter = 0;
    int heavyFighter = 1;
    std::vector<SpecShipBlockDescriptor> fleet1;
    std::vector<SpecShipBlockDescriptor> fleet2;
    SpecShipBlockDescriptor lFighter;
    lFighter.count = 900000;
    lFighter.typeIndex = lightFighter;
    SpecShipBlockDescriptor hFighter;
    hFighter.count = 100000;
    hFighter.typeIndex = heavyFighter;
    fleet1.push_back(lFighter);
    fleet1.push_back(hFighter);
    fleet2.push_back(lFighter);
    fleet2.push_back(hFighter);
    int gpuIndex = 0;
    Simulator simulator(gpuIndex, randomSeed, randomSeed2);
    simulator.simulate(fleet1, fleet2);
    return 0;
}

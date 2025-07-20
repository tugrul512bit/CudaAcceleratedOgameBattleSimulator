#include <iostream>
#include <vector>
#include <map>
#include <algorithm>
#include <memory>
#include <string>
#include "kernels.cuh"



struct GpuStream {
    int deviceIndex;
    cudaStream_t stream;
    GpuStream(int deviceIndexPrm = 0) {
        deviceIndex = deviceIndexPrm;
        cudaSetDevice(deviceIndex);
        cudaStreamCreate(&stream);
    }
    ~GpuStream() {
        cudaSetDevice(deviceIndex);
        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
    }
};

struct GpuBuffer {
    char* ptr_d;
    char* ptr_h;
    uint64_t numBytes;
    std::shared_ptr<GpuStream> gpuStream;
    GpuBuffer(uint64_t numBytesPrm = 0, std::shared_ptr<GpuStream> streamPrm = nullptr, bool isScalar = false) {
        gpuStream = streamPrm;
        numBytes = numBytesPrm;
        if (numBytes > 0 && streamPrm != nullptr) {
            cudaSetDevice(gpuStream->deviceIndex);
            cudaMallocAsync(&ptr_d, numBytes, gpuStream->stream);
            cudaMemsetAsync(ptr_d, 0, numBytes, gpuStream->stream);
            cudaMallocHost(&ptr_h, numBytes);
        }
    }
    template<typename TYPE>
    TYPE get(uint32_t index) {
        return reinterpret_cast<TYPE*>(ptr_h)[index];
    }
    template<typename TYPE>
    void set(uint32_t index, TYPE value) {
        reinterpret_cast<TYPE*>(ptr_h)[index] = value;
    }
    void updateDevice() {
        cudaSetDevice(gpuStream->deviceIndex);
        cudaMemcpyAsync(ptr_d, ptr_h, numBytes, cudaMemcpyHostToDevice, gpuStream->stream);
        cudaStreamSynchronize(gpuStream->stream);
    }
    void updateHost() {
        cudaSetDevice(gpuStream->deviceIndex);
        cudaMemcpyAsync(ptr_h, ptr_d, numBytes, cudaMemcpyHostToDevice, gpuStream->stream);
        cudaStreamSynchronize(gpuStream->stream);
    }
    ~GpuBuffer() {
        if (numBytes > 0) {
            cudaSetDevice(gpuStream->deviceIndex);
            cudaFreeAsync(ptr_d, gpuStream->stream);
            cudaFreeHost(ptr_h);
        }
    }
};

struct GpuKernel {
    std::shared_ptr<GpuStream> gpuStream;
    std::vector<std::shared_ptr<GpuBuffer>> gpuParameters;
    std::vector<void*> args;
    void* ptr;
    GpuKernel(void* kernel, std::vector<std::shared_ptr<GpuBuffer>> parameters, std::shared_ptr<GpuStream> stream) {
        ptr = kernel;
        gpuStream = stream;
        gpuParameters = parameters;
        for (auto& parameter : gpuParameters) {
            args.push_back(&parameter->ptr_d);
        }
    }
    void run(uint32_t numBlocks, uint32_t numThreads) {
        cudaSetDevice(gpuStream->deviceIndex);
        cudaLaunchKernel(ptr, dim3(numBlocks, 1, 1), dim3(numThreads, 1, 1), args.data(), 0, gpuStream->stream);
    }
};

struct Gpu {
    std::shared_ptr<GpuStream> gpuStream;
    std::map<std::string, std::shared_ptr<GpuBuffer>> gpuBuffer;
    std::map<std::string, std::shared_ptr<GpuKernel>> gpuKernel;
    uint32_t maxBlocks;
    Gpu(int gpuIndex = -1) {
        if (gpuIndex >= 0) {
            gpuStream = std::make_shared<GpuStream>(gpuIndex);
            cudaDeviceProp deviceProperties;
            cudaSetDevice(gpuStream->deviceIndex);
            cudaGetDeviceProperties(&deviceProperties, gpuStream->deviceIndex);
            maxBlocks = deviceProperties.maxBlocksPerMultiProcessor * deviceProperties.multiProcessorCount;
        }
    }
    void addBuffer(std::string name, uint64_t numBytes) {
        gpuBuffer[name] = std::make_shared<GpuBuffer>(numBytes, gpuStream);
    }
    void addKernel(std::string kernelName, void* kernelPtr, std::vector<std::string> parameterNames) {
        std::vector<std::shared_ptr<GpuBuffer>> parameters;
        for (auto& prm : parameterNames) {
            auto& parameter = gpuBuffer[prm];
            parameters.push_back(parameter);
        }
        gpuKernel[kernelName] = std::make_shared<GpuKernel>(kernelPtr, parameters, gpuStream);
    }
    void runKernel(std::string name, uint32_t numTotalThreads) {
        uint32_t numThreadsPerBlock = 512;
        uint32_t required = (numTotalThreads + numThreadsPerBlock - 1) / numThreadsPerBlock;
        uint32_t utilized = maxBlocks < required ? maxBlocks : required;
        gpuKernel[name]->run(utilized, numThreadsPerBlock);
    }
    void wait() {
        cudaSetDevice(gpuStream->deviceIndex);
        cudaStreamSynchronize(gpuStream->stream);
    }

};

// This is only a wrapper for multiple gpus to compute same thing as a simulation-level-parallelism.
struct GpuSystem {
    int count;
    std::vector<std::shared_ptr<Gpu>> gpus;
    GpuSystem(bool debugOutput = false) {
        cudaGetDeviceCount(&count);
        for (int i = 0; i < count; i++) {
            if (debugOutput) {
                cudaDeviceProp properties;
                cudaGetDeviceProperties(&properties, i);
                std::cout << "Creating device object for " << properties.name << std::endl;
            }
            gpus.push_back(std::make_shared<Gpu>(i));
        }
    }

    template<typename TYPE>
    void addBuffer(std::string name, uint32_t numElements) {
        for (auto& gpu : gpus) {
            gpu->addBuffer(name, numElements * sizeof(TYPE));
        }
    }
    template<typename TYPE>
    void writeToBuffer(std::string name, uint32_t index, TYPE value) {
        for (auto& gpu : gpus) {
            gpu->gpuBuffer[name]->set<TYPE>(index, value);
        }
    }
    template<typename TYPE>
    void readFromBuffer(std::string name, uint32_t index) {
        for (auto& gpu : gpus) {
            gpu->gpuBuffer[name]->get<TYPE>(index);
        }
    }
    void updateDeviceBuffer(std::string name) {
        for (auto& gpu : gpus) {
            gpu->gpuBuffer[name]->updateDevice();
        }
    }
    void updateHostBuffer(std::string name) {
        for (auto& gpu : gpus) {
            gpu->gpuBuffer[name]->updateHost();
        }
    }
    void addKernel(std::string kernelName, void* kernelPtr, std::vector<std::string> parameterNames) {
        for (auto& gpu : gpus) {
            gpu->addKernel(kernelName, kernelPtr, parameterNames);
        }
    }
    void runKernel(std::string name, uint32_t numTotalThreads) {
        for (auto& gpu : gpus) {
            gpu->runKernel(name, numTotalThreads);
        }
    }
    void wait() {
        for (auto& gpu : gpus) {
            gpu->wait();
        }
    }

};

struct Simulator {
    std::shared_ptr<GpuSystem> system;
    uint32_t numShips[2];
    uint64_t globalBaseRandomSeed[2];
    int numShipTypes;
    Simulator(bool debug = false, int randomSeedTeam1 = 0, int randomSeedTeam2 = 100) {
        system = std::make_shared<GpuSystem>(debug);
        if (debug) {
            std::cout << "num gpus = " << system->count << std::endl;
        }
        system->addBuffer<uint32_t>("team 1 number of ships", 1);
        system->addBuffer<uint32_t>("team 2 number of ships", 1);
        system->addBuffer<uint64_t>("team 1 global base random seed", 1);
        system->addBuffer<uint64_t>("team 2 global base random seed", 1);
        numShips[0] = 0;
        numShips[1] = 0;
        globalBaseRandomSeed[0] = randomSeedTeam1;
        globalBaseRandomSeed[1] = randomSeedTeam2;
        system->writeToBuffer("team 1 number of ships", 0, numShips[0]);
        system->writeToBuffer("team 2 number of ships", 0, numShips[1]);
        system->writeToBuffer("team 1 global base random seed", 0, globalBaseRandomSeed[0]);
        system->writeToBuffer("team 2 global base random seed", 0, globalBaseRandomSeed[1]);

        // Adding ship specs.
        std::vector<uint32_t> shipTypeIndex;
        std::vector<uint32_t> shipOffense;
        std::vector<uint32_t> shipShield;
        std::vector<uint32_t> shipHull;
        numShipTypes = 0;
        // Light fighter.
        {
            shipTypeIndex.push_back(numShipTypes);
            shipOffense.push_back(50);
            shipShield.push_back(10);
            shipHull.push_back(400);
            numShipTypes++;
        }
        // Heavy fighter.
        {
            shipTypeIndex.push_back(numShipTypes);
            shipOffense.push_back(150);
            shipShield.push_back(25);
            shipHull.push_back(1000);
            numShipTypes++;
        }
        system->addBuffer<uint32_t>("default ship type index", numShipTypes);
        system->addBuffer<uint32_t>("default ship offense", numShipTypes);
        system->addBuffer<uint32_t>("default ship shield", numShipTypes);
        system->addBuffer<uint32_t>("default ship hull", numShipTypes);
        system->addBuffer<uint32_t>("number of ship types", 1);
        system->writeToBuffer<uint32_t>("number of ship types", 0, numShipTypes);
        system->updateDeviceBuffer("number of ship types");
        for (int i = 0; i < numShipTypes; i++) {
            system->writeToBuffer<uint32_t>("default ship type index", i, shipTypeIndex[i]);
            system->writeToBuffer<uint32_t>("default ship offense", i, shipOffense[i]);
            system->writeToBuffer<uint32_t>("default ship shield", i, shipShield[i]);
            system->writeToBuffer<uint32_t>("default ship hull", i, shipHull[i]);
        }
        system->updateDeviceBuffer("default ship type index");
        system->updateDeviceBuffer("default ship offense");
        system->updateDeviceBuffer("default ship shield");
        system->updateDeviceBuffer("default ship hull");

        //system->addBuffer
    }
    // team: 1 or 2
    void addShips(uint32_t totalNumberOfShips, int team) {
        numShips[team - 1] = totalNumberOfShips;
        std::string teamString = std::string("team ")+std::to_string(team)+std::string(" ");
        // Init random seed.
        system->addBuffer<curandState>(teamString + std::string("random seed"), numShips[team - 1]);
        // Updating new number of ships on device.
        system->writeToBuffer(teamString+std::string("number of ships"), 0, numShips[team - 1]);
        system->updateDeviceBuffer(teamString + std::string("number of ships"));
        // Initializing random seeds of each ship.
        system->addKernel(teamString + std::string("init random seed"), (void*)Kernels::k_initializeRandomSeeds, { teamString + std::string("random seed"), teamString + "number of ships", teamString + "global base random seed" });
        system->runKernel(teamString + std::string("init random seed"), numShips[team - 1]);

        system->addBuffer<SpaceShip>(teamString + std::string("ships"), numShips[team - 1]);
        system->wait();
    }
    void initShips() {
        system->addKernel(std::string("init space ships"), (void*)Kernels::k_initializeShipHulls, {
            std::string("team 1 number of ships"),
            std::string("team 2 number of ships"),
            std::string("team 1 ships"),
            std::string("team 2 ships"),
            std::string("default ship hull"),
            std::string("number of ship types"),
        });
        uint32_t n = (numShips[0] < numShips[1]) ? numShips[1] : numShips[0];
        system->runKernel(std::string("init space ships"), n);
        system->wait();
    }

    void demo() {
        // Adding 1000 ships of type 0.
        int team1 = 1;
        int team2 = 2;
        addShips(1000, team1);
        addShips(1000, team2);
        initShips();
    }
};

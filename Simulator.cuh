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
    void copyTo(GpuBuffer* destination, size_t numBytes) {
        cudaSetDevice(gpuStream->deviceIndex);
        cudaMemcpyAsync(destination->ptr_d, ptr_d, numBytes, cudaMemcpyDeviceToDevice, gpuStream->stream);
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
    template<typename TYPE>
    void sort() {
        cudaSetDevice(gpuStream->deviceIndex);
        thrust::device_ptr<TYPE> begin_d((TYPE*)(ptr_d));
        thrust::device_ptr<TYPE> end_d((TYPE*)(ptr_d + numBytes));
        thrust::sort(thrust::cuda::par.on(gpuStream->stream), begin_d, end_d);
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
        else {
            gpuStream = nullptr;
            maxBlocks = 0;
        }
    }
    
    template<typename TYPE>
    void addBuffer(std::string name, uint64_t numElements) {
        gpuBuffer[name] = std::make_shared<GpuBuffer>(numElements * sizeof(TYPE), gpuStream);
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

struct Simulator {
    std::shared_ptr<Gpu> gpu;
    uint32_t numShips[2];
    uint64_t globalBaseRandomSeed[2];
    int numShipTypes;
    Simulator(int gpuIndex = 0, int randomSeedTeam1 = 0, int randomSeedTeam2 = 100) {
        gpu = std::make_shared<Gpu>(gpuIndex);

        
        gpu->addBuffer<uint32_t>("team 1 number of ships", 1);
        gpu->addBuffer<uint32_t>("team 2 number of ships", 1);
        gpu->addBuffer<uint32_t>("team 1 number of ships backup", 1);
        gpu->addBuffer<uint32_t>("team 2 number of ships backup", 1);
        gpu->addBuffer<uint64_t>("team 1 global base random seed", 1);
        gpu->addBuffer<uint64_t>("team 2 global base random seed", 1);
        numShips[0] = 0;
        numShips[1] = 0;
        globalBaseRandomSeed[0] = randomSeedTeam1;
        globalBaseRandomSeed[1] = randomSeedTeam2;
        gpu->gpuBuffer["team 1 number of ships"]->set<uint32_t>(0, numShips[0]);
        gpu->gpuBuffer["team 2 number of ships"]->set<uint32_t>(0, numShips[1]);
        gpu->gpuBuffer["team 1 global base random seed"]->set<uint64_t>(0, globalBaseRandomSeed[0]);
        gpu->gpuBuffer["team 2 global base random seed"]->set<uint64_t>(0, globalBaseRandomSeed[1]);
        gpu->gpuBuffer["team 1 number of ships"]->updateDevice();
        gpu->gpuBuffer["team 2 number of ships"]->updateDevice();
        gpu->gpuBuffer["team 1 global base random seed"]->updateDevice();
        gpu->gpuBuffer["team 2 global base random seed"]->updateDevice();
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
        gpu->addBuffer<uint32_t>("default ship type index", numShipTypes);
        gpu->addBuffer<uint32_t>("default ship offense", numShipTypes);
        gpu->addBuffer<uint32_t>("default ship shield", numShipTypes);
        gpu->addBuffer<uint32_t>("default ship hull", numShipTypes);
        gpu->addBuffer<uint32_t>("number of ship types", 1);
        gpu->gpuBuffer["number of ship types"]->set<uint32_t>(0, numShipTypes);
        gpu->gpuBuffer["number of ship types"]->updateDevice();
        for (int i = 0; i < numShipTypes; i++) {
            gpu->gpuBuffer["default ship type index"]->set<uint32_t>(i, shipTypeIndex[i]);
            gpu->gpuBuffer["default ship offense"]->set<uint32_t>(i, shipOffense[i]);
            gpu->gpuBuffer["default ship shield"]->set<uint32_t>(i, shipShield[i]);
            gpu->gpuBuffer["default ship hull"]->set<uint32_t>(i, shipHull[i]);
        }
        gpu->gpuBuffer["default ship type index"]->updateDevice();
        gpu->gpuBuffer["default ship offense"]->updateDevice();
        gpu->gpuBuffer["default ship shield"]->updateDevice();
        gpu->gpuBuffer["default ship hull"]->updateDevice();

        //system->addBuffer
    }
    // team: 1 or 2
    void addShips(int team, std::vector<SpecShipBlockDescriptor> ships) {
        uint32_t totalNumberOfShips = 0;
        for (auto& c : ships) {
            totalNumberOfShips += c.count;
        }
        numShips[team - 1] = totalNumberOfShips;
        std::string teamString = std::string("team ") + std::to_string(team) + std::string(" "); 
        // Init random seed.
        gpu->addBuffer<curandState>(teamString + std::string("random seed"), numShips[team - 1]);
        // Updating new number of ships on device.
        gpu->gpuBuffer[teamString + std::string("number of ships")]->set<uint32_t>(0, numShips[team - 1]);
        gpu->gpuBuffer[teamString + std::string("number of ships")]->updateDevice();
        // Initializing random seeds of each ship.
        gpu->addKernel(teamString + std::string("init random seed"), (void*)Kernels::k_initializeRandomSeeds, { teamString + std::string("random seed"), teamString + "number of ships", teamString + "global base random seed" });
        gpu->runKernel(teamString + std::string("init random seed"), numShips[team - 1]);
        gpu->addBuffer<SpaceShip>(teamString + std::string("ships"), numShips[team - 1]);
        gpu->addBuffer<SpaceShip>(teamString + std::string("ships sorted"), numShips[team - 1]);
        gpu->addBuffer<uint32_t>(teamString + std::string("target index reduced"), numShips[team - 1]);
        gpu->addBuffer<uint32_t>(teamString + std::string("damage reduced"), numShips[team - 1]);
        gpu->addBuffer<SpaceShip>(teamString + std::string("ships backup"), numShips[team - 1]);
        gpu->gpuBuffer[teamString + std::string("number of ships backup")]->set<uint32_t>(0, numShips[team - 1]);
        gpu->gpuBuffer[teamString + std::string("number of ships backup")]->updateDevice();
        gpu->addBuffer<uint32_t>(teamString + std::string("number of descriptors"), 1);
        gpu->gpuBuffer[teamString + std::string("number of descriptors")]->set<uint32_t>(0, ships.size());
        gpu->addBuffer<SpecShipBlockDescriptor>(teamString + std::string("ship type descriptors"), ships.size());
        for (int i = 0; i < ships.size(); i++) {
            gpu->gpuBuffer[teamString + std::string("ship type descriptors")]->set<SpecShipBlockDescriptor>(i, ships[i]);
        }
        gpu->gpuBuffer[teamString + std::string("number of descriptors")]->updateDevice();
        gpu->gpuBuffer[teamString + std::string("ship type descriptors")]->updateDevice();
        gpu->wait();
    }
    void initShips() {
        gpu->addKernel(std::string("init space ships"), (void*)Kernels::k_initializeShipHulls, {
            std::string("team 1 number of ships"),
            std::string("team 2 number of ships"),
            std::string("team 1 ships"),
            std::string("team 2 ships"),
            std::string("team 1 ships backup"),
            std::string("team 2 ships backup"),
            std::string("team 1 ship type descriptors"),
            std::string("team 2 ship type descriptors"),
            std::string("team 1 number of descriptors"),
            std::string("team 2 number of descriptors"),
            std::string("default ship hull"),
            std::string("number of ship types")
            });
        uint32_t n = (numShips[0] < numShips[1]) ? numShips[1] : numShips[0];
        gpu->runKernel(std::string("init space ships"), n);
        gpu->wait();
    }
    void pickTargetsKernelInit() {
        gpu->addKernel(std::string("pick targets"), (void*)Kernels::k_pickTarget, { 
            std::string("team 1 random seed"),  
            std::string("team 2 random seed"),
            std::string("team 1 ships"),
            std::string("team 2 ships"),
            std::string("team 1 number of ships"),
            std::string("team 2 number of ships"),
        });
    }
    void pickTargets() {
        numShips[0] = gpu->gpuBuffer["team 1 number of ships backup"]->get<uint32_t>(0);
        numShips[1] = gpu->gpuBuffer["team 2 number of ships backup"]->get<uint32_t>(0);
        uint32_t n = (numShips[0] < numShips[1]) ? numShips[1] : numShips[0];
        gpu->runKernel(std::string("pick targets"), n);
        gpu->wait();
    }
    void sortShipsOnTargets() {
       gpu->gpuBuffer["team 1 ships"]->copyTo(gpu->gpuBuffer["team 1 ships sorted"].get(), gpu->gpuBuffer["team 1 ships"]->numBytes);
       gpu->gpuBuffer["team 2 ships"]->copyTo(gpu->gpuBuffer["team 2 ships sorted"].get(), gpu->gpuBuffer["team 2 ships"]->numBytes);
       gpu->gpuBuffer["team 1 ships sorted"]->sort<SpaceShip>();
       gpu->gpuBuffer["team 2 ships sorted"]->sort<SpaceShip>();
       gpu->wait();
    }
    void reduceSegmentDamageByTargetIndex(int team) {
        printf("\n --------- %i ---------- \n", numShips[team - 1]);
        std::string teamString = std::string("team ") + std::to_string(team) + " ";
        auto keysBegin = thrust::make_transform_iterator((SpaceShip*)gpu->gpuBuffer[teamString + "ships sorted"]->ptr_d, [] __device__(const SpaceShip & ship) -> uint32_t { return ship.targetIndex; });
        LutHelper helper;
        helper.offensePtr = (uint32_t*) gpu->gpuBuffer["default ship offense"]->ptr_d;
        auto valsBegin = thrust::make_transform_iterator((SpaceShip*)gpu->gpuBuffer[teamString + "ships sorted"]->ptr_d, helper);

        thrust::device_ptr<uint32_t> outKeys((uint32_t*)(gpu->gpuBuffer[teamString + "target index reduced"]->ptr_d));
        thrust::device_ptr<uint32_t> outVals((uint32_t*)(gpu->gpuBuffer[teamString + "damage reduced"]->ptr_d));
        auto newEnd = thrust::reduce_by_key(
            thrust::cuda::par.on(gpu->gpuStream->stream),
            keysBegin, keysBegin + numShips[team - 1],
            valsBegin,
            outKeys,
            outVals,
            thrust::equal_to<uint32_t>(),
            thrust::plus<uint32_t>()
        );

        int numSegments = newEnd.first - outKeys;
        // Print results
        for (int i = 0; i < numSegments; i++) {
            std::cout << "Ship index: " << outKeys[i]
                << " total damage = " << outVals[i] << "\n";
        }
    }
    void simulate(std::vector<SpecShipBlockDescriptor> fleet1, std::vector<SpecShipBlockDescriptor> fleet2) {
        int team1 = 1;
        int team2 = 2;
        addShips(team1, fleet1);
        addShips(team2, fleet2);
        initShips();
        pickTargetsKernelInit();
        pickTargets();
        sortShipsOnTargets();
        reduceSegmentDamageByTargetIndex(team1);
        reduceSegmentDamageByTargetIndex(team2);
    }
};

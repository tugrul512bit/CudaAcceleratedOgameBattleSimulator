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
    Gpu(int gpuIndex = -1) {
        if (gpuIndex >= 0) {
            gpuStream = std::make_shared<GpuStream>(gpuIndex);
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
        uint32_t numBlocks = (numTotalThreads + numThreadsPerBlock - 1) / numThreadsPerBlock;
        gpuKernel[name]->run(numBlocks, numThreadsPerBlock);
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
    std::vector<uint32_t> shipTypeIndex;
    uint32_t numShips;
    Simulator(bool debug = false) {
        system = std::make_shared<GpuSystem>(debug);
        if (debug) {
            std::cout << "num gpus = " << system->count << std::endl;
        }
        system->addBuffer<uint32_t>("number of ships", 1);
        numShips = 0;
        system->writeToBuffer("number of ships", 0, numShips);
    }
    void addShips(uint32_t numberOfNewShips, uint32_t typeIndexOfNewShips) {
        shipTypeIndex.resize(shipTypeIndex.size() + numberOfNewShips);
        for (int i = 0; i < numberOfNewShips; i++) {
            shipTypeIndex[i + numShips] = typeIndexOfNewShips;
        }
        numShips += numberOfNewShips;
        system->addBuffer<uint32_t>("ship type index", numShips);
        system->writeToBuffer("number of ships", 0, numShips);
        system->wait();
    }
};

#include "kernels.cuh"
#include "stdio.h"
namespace Kernels {
	__global__ void k_initializeShipIndex(uint32_t* numShips, uint32_t* shipIndex_d) {
		uint32_t n = numShips[0];
		uint32_t threadIndex = threadIdx.x + blockIdx.x * blockDim.x;
		uint32_t numThreads = blockDim.x * gridDim.x;
		uint32_t numSteps = (n + numThreads - 1) / numThreads;
		for (uint32_t i = 0; i < numSteps; i++) {
			uint32_t item = i * numThreads + threadIndex;
			if (item < n) {
                shipIndex_d[item] = item; printf("%i ", item);
            }
		}
	}
}
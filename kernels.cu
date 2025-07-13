#include "kernels.cuh"
#include "stdio.h"
namespace Kernels {
	__global__ void k_initializeRandomSeeds(curandState* seeds, uint32_t* numShips, uint64_t* baseSeed) {
		uint32_t n = numShips[0];
		uint64_t commonSeed = baseSeed[0];
		uint32_t threadIndex = threadIdx.x + blockIdx.x * blockDim.x;
		uint32_t numThreads = blockDim.x * gridDim.x;
		uint32_t numSteps = (n + numThreads - 1) / numThreads;
		for (uint32_t i = 0; i < numSteps; i++) {
			uint32_t item = i * numThreads + threadIndex;
			if (item < n) {
				curand_init(commonSeed, item, 0, &seeds[item]);
			}
		}
	}
}

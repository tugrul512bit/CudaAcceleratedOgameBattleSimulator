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
	__global__ void k_initializeShipHulls(uint32_t* numShipsTeam1, uint32_t* numShipsTeam2, SpaceShip* shipsTeam1, SpaceShip* shipsTeam2, uint32_t* defaultShipHulls, uint32_t* numberOfShipTypes) {
		uint32_t n1 = numShipsTeam1[0];
		uint32_t n2 = numShipsTeam2[0];
		uint32_t n = (n1 < n2) ? n2 : n1;
		uint32_t h = numberOfShipTypes[0];
		uint32_t threadIndex = threadIdx.x + blockIdx.x * blockDim.x;
		uint32_t numThreads = blockDim.x * gridDim.x;
		uint32_t numSteps = (n + numThreads - 1) / numThreads;
		__shared__ uint32_t defaultHulls[50]; 
		if (threadIdx.x < h) {
			defaultHulls[threadIdx.x] = defaultShipHulls[threadIdx.x]; 
        }
		__syncthreads();
		for (uint32_t i = 0; i < numSteps; i++) {
			uint32_t item = i * numThreads + threadIndex;
			if (item < n1) {
				shipsTeam1[item].remainingHull = defaultHulls[shipsTeam1[item].typeIndex]; 
			}
			if (item < n2) {
				shipsTeam2[item].remainingHull = defaultHulls[shipsTeam2[item].typeIndex];
			}
		}
	}
}

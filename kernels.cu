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
	__global__ void k_initializeShipHulls(uint32_t* numShipsTeam1, uint32_t* numShipsTeam2, SpaceShip* shipsTeam1, SpaceShip* shipsTeam2, SpaceShip* shipsBackupTeam1, SpaceShip* shipsBackupTeam2, SpecShipBlockDescriptor* shipTypesAndNumbersTeam1, SpecShipBlockDescriptor* shipTypesAndNumbersTeam2, uint32_t* numDescriptorsTeam1, uint32_t* numDescriptorsTeam2, uint32_t* defaultShipHulls, uint32_t* numberOfShipTypes) {
		uint32_t n1 = numShipsTeam1[0];
		uint32_t n2 = numShipsTeam2[0];
		uint32_t d1 = numDescriptorsTeam1[0];
		uint32_t d2 = numDescriptorsTeam2[0];
		uint32_t n = (n1 < n2) ? n2 : n1;
		uint32_t h = numberOfShipTypes[0];
		uint32_t threadIndex = threadIdx.x + blockIdx.x * blockDim.x;
		uint32_t numThreads = blockDim.x * gridDim.x;
		uint32_t numSteps = (n + numThreads - 1) / numThreads;
		__shared__ uint32_t defaultHulls[10];
		__shared__ uint32_t typeOffsetTeam1[11];
		__shared__ uint32_t typeOffsetTeam2[11];
		__shared__ uint32_t typeTeam1[11];
		__shared__ uint32_t typeTeam2[11];
		if (threadIdx.x < h) {
			defaultHulls[threadIdx.x] = defaultShipHulls[threadIdx.x];
		}
		uint32_t currentTeam1Offset = 0;
		uint32_t currentTeam2Offset = 0;
		if (threadIdx.x == 0) {
			for (int i = 0; i < d1; i++) {
				typeTeam1[i] = shipTypesAndNumbersTeam1[i].typeIndex;
				typeOffsetTeam1[i] = currentTeam1Offset;
				currentTeam1Offset += shipTypesAndNumbersTeam1[i].count; 
			}
			typeOffsetTeam1[d1] = currentTeam1Offset;
			for (int i = 0; i < d2; i++) {
				typeTeam2[i] = shipTypesAndNumbersTeam2[i].typeIndex;
				typeOffsetTeam2[i] = currentTeam2Offset;
				currentTeam2Offset += shipTypesAndNumbersTeam2[i].count;
			}
			typeOffsetTeam2[d1] = currentTeam2Offset;
		}
		__syncthreads();
		for (uint32_t i = 0; i < numSteps; i++) {
			uint32_t item = i * numThreads + threadIndex;
			if (item < n1) {
				for (int j = 0; j < d1; j++) {
					if (item >= typeOffsetTeam1[j] && item < typeOffsetTeam1[j + 1]) {
						uint32_t hull = defaultHulls[typeTeam1[j]];
						shipsTeam1[item].remainingHull = hull;
						shipsBackupTeam1[item].remainingHull = hull;
						break;
					}
				}
			}
			if (item < n2) {
				for (int j = 0; j < d2; j++) {
					if (item >= typeOffsetTeam2[j] && item < typeOffsetTeam2[j + 1]) {
						uint32_t hull = defaultHulls[typeTeam2[j]];
				        shipsTeam2[item].remainingHull = hull;
						shipsBackupTeam2[item].remainingHull = hull;
				        break;
					}
				}
			}
		}
	}
}

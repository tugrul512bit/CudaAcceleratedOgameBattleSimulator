#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
struct SpaceShip {
	uint32_t typeIndex;
	uint32_t remainingHull;
	uint32_t targetIndex;
};
struct SpecShipBlockDescriptor {
	uint32_t typeIndex;
	uint32_t count;
};
namespace Kernels {
	__global__ void k_initializeRandomSeeds(curandState* seeds, uint32_t* numShips, uint64_t* baseSeed);
	__global__ void k_initializeShipHulls(uint32_t* numShipsTeam1, uint32_t* numShipsTeam2, SpaceShip* shipsTeam1, SpaceShip* shipsTeam2, SpaceShip* shipsBackupTeam1, SpaceShip* shipsBackupTeam2, SpecShipBlockDescriptor* shipTypesAndNumbersTeam1, SpecShipBlockDescriptor* shipTypesAndNumbersTeam2, uint32_t* numDescriptorsTeam1, uint32_t* numDescriptorsTeam2, uint32_t* defaultShipHulls, uint32_t* numberOfShipTypes);
}

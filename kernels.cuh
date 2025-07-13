#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
namespace Kernels {
	__global__ void k_initializeShipIndex(uint32_t* numShips, uint32_t* shipIndex_d);
}

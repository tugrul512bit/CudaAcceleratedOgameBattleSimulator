#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
namespace Kernels {
	__global__ void k_initializeRandomSeeds(curandState* seeds, uint32_t* numShips, uint64_t* baseSeed);
}

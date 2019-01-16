#include "cuda_helper.h"


// vec = f
__global__ void initialize_float(float* vec, int size, float f) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	if (id < size)
		vec[id] = f;
}
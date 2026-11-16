#include <cuda_runtime.h>

#include <iostream>

namespace {

constexpr int expected_value = 42;

__global__ void write_value(int* value)
{
    *value = expected_value;
}

bool check(cudaError_t result, const char* operation)
{
    if (result == cudaSuccess) {
        return true;
    }
    std::cerr << operation << ": " << cudaGetErrorString(result) << '\n';
    return false;
}

} // namespace

int main()
{
    int* value = nullptr;
    if (!check(cudaMallocManaged(&value, sizeof(int)), "Allocate CUDA buffer")) {
        return 1;
    }

    *value = 0;
    write_value<<<1, 1>>>(value);
    const bool ran = check(cudaGetLastError(), "Launch CUDA kernel") &&
                     check(cudaDeviceSynchronize(), "Wait for CUDA kernel");
    const int observed = ran ? *value : 0;
    const bool freed = check(cudaFree(value), "Free CUDA buffer");

    if (!ran || !freed) {
        return 1;
    }
    if (observed != expected_value) {
        std::cerr << "Expected " << expected_value << ", got " << observed << '\n';
        return 1;
    }
    std::cout << "CUDA smoke test passed (GPU wrote " << observed << ").\n";
    return 0;
}

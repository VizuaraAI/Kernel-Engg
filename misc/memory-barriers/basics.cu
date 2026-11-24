/**
An introductory example on asynchronous barriers in CUDA C++.
The following example simulates the behavior of __syncthreads() using barriers.

A barrier is a synchronization primitive that counts the number of arrivals it receives
from different threads. This count is asynchronous in nature and does not block any thread.
The thread that calls barrier.wait() is the one that is blocked until current arrival counts equals
the expected arrival count provided during initialization.

A cuda::barrier is initialized as follows:

@code
#include <cuda/barrier>

int expectedArrivalCount = 32
init(&bar, expectedArrivalCount);
@endcode

The barrier's counter is initialized with a count of `expectedArrivalCount`.

When a thread hits the barrier, it must call the arrive() function:

@code
barrier_t::arrival_token token = bar.arrive();
@endcode

Calling `bar.arrive()` decrements the count of the barrier by one. In our case, as the
barrier was initialized with a count of 32, its count after calling `bar.arrive()` will
become 31. The `arrival_token` returns by barrier.arrive() is associated with the current `phase`
of the barrier. When all 32 threads will call `bar.arrive()` and the counter inside the barrier
becomes 0 then parity bit of the barrier is changed (ex. changing the parity from 0 to 1):

@code
barrier.wait(std::move(token))
@endcode

The `token` holds the parity of the barrier. `barrier.wait` will block the current thread
until the count of arrived threads matches the expectedArrivalCount on a barrier with the same
parity as `token`.

Once the barrier arrival count reaches zero, it switches the parity bit and resets the counter
to `expectedArrivalCount`.

Docs:
https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-barriers.html
*/
#include <algorithm>
#include <cooperative_groups.h>
#include <cuda/barrier>
#include <cuda_runtime.h>
#include <iostream>

using barrier_t = cuda::barrier<cuda::thread_scope_block>;

__global__ void barrierExampleKernel() {
    __shared__ cuda::barrier<cuda::thread_scope_block> bar;
    __shared__ float sharedMemoryRegion[32];
    auto currentThreadBlock = cooperative_groups::this_thread_block();

    if (currentThreadBlock.thread_rank() == 0) {
        // initialize the memory barrier only once with expected
        // arrival count of 32 transactions. The arrival count is initialized
        // as 32.
        init(&bar, 32);
    }
    currentThreadBlock.sync();

    sharedMemoryRegion[threadIdx.x] = static_cast<float>(threadIdx.x) * 2 / 3;
    printf("Thread %d wrote to SMEM\n", threadIdx.x);

    // the current thread registers its arrival with the memory
    // barrier (decrementing the arrival count by 1)
    barrier_t::arrival_token token1 = bar.arrive();
    // Wait for all threads participating in the barrier to complete bar.arrive().
    // Meaning, wait until the arrival count becomes 0.
    bar.wait(std::move(token1));

    // Once wait() completes, the barrier is reset and its count is initialized to 32.

    float sharedMemoryValue = sharedMemoryRegion[threadIdx.x] * 3 / 2;
    printf("Thread %d read from SMEM\n", threadIdx.x);

    // the current thread registers its arrival with the memory
    // barrier (decrementing the arrival count by 1)
    // bar.arrive_and_wait() is equivalent to
    // bar.wait(bar.arrive())
    bar.arrive_and_wait();
}

int main() {
    dim3 threadBlockSize(32, 1, 1);
    dim3 gridSize(1, 1, 1);
    barrierExampleKernel<<<gridSize, threadBlockSize>>>();
    cudaDeviceSynchronize();
    return 0;
}
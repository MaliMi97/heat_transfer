#ifndef __REDUCTION_H__
#define __REDUCTION_H__

#include "traits.h"
#include "mesh.h"
#include "solver.h"

/**
* Number of threads per block for reduction.
*/
const int reductionThreadsPerBlock = 256;

/**
* Kernel for reduction, which is used to count the norm.
*/
template< typename Traits, const int threadsPerBlock>
__global__ void global_reduction( typename Traits::real* dev_output, typename Traits::real* dev_input, typename Traits::idx size)
{
    using idx = typename Traits::idx;
    using real = typename Traits::real;

    extern __shared__ volatile real sdata[];

    idx tid = threadIdx.x;
    idx gid = blockIdx.x*threadsPerBlock*2 + threadIdx.x;
    idx gridSize = threadsPerBlock*2*gridDim.x;
    sdata[tid] = 0;
    while (gid < size)
    {
        sdata[tid] = sdata[tid] > dev_input[gid] ? sdata[tid] : dev_input[gid];
        sdata[tid] = sdata[tid] > dev_input[gid+threadsPerBlock] ? sdata[tid] : dev_input[gid+threadsPerBlock];
        gid += gridSize;
    }
    __syncthreads();

    if (threadsPerBlock == 1024)
    {
        if (tid < 512)
            sdata[tid] = sdata[tid] > sdata[tid+512] ? sdata[tid] : sdata[tid+512];
        __syncthreads();
    }
    if (threadsPerBlock >= 512)
    {
        if (tid < 256)
            sdata[tid] = sdata[tid] > sdata[tid+256] ? sdata[tid] : sdata[tid+256];
        __syncthreads();
    }
    if (threadsPerBlock >= 256)
    {
        if (tid < 128)
            sdata[tid] = sdata[tid] > sdata[tid+128] ? sdata[tid] : sdata[tid+128];
        __syncthreads();
    }
    if (threadsPerBlock >= 128)
    {
        if (tid < 64)
            sdata[tid] = sdata[tid] > sdata[tid+64] ? sdata[tid] : sdata[tid+64];
        __syncthreads();
    }
    if (tid < 32)
    {
        if (threadsPerBlock >= 64)
            sdata[tid] = sdata[tid] > sdata[tid+32] ? sdata[tid] : sdata[tid+32];
        if (threadsPerBlock >= 32)
            sdata[tid] = sdata[tid] > sdata[tid+16] ? sdata[tid] : sdata[tid+16];
        if (threadsPerBlock >= 16)
            sdata[tid] = sdata[tid] > sdata[tid+8] ? sdata[tid] : sdata[tid+8];
        if (threadsPerBlock >= 8)
            sdata[tid] = sdata[tid] > sdata[tid+4] ? sdata[tid] : sdata[tid+4];
        if (threadsPerBlock >= 4)
            sdata[tid] = sdata[tid] > sdata[tid+2] ? sdata[tid] : sdata[tid+2];
        if (threadsPerBlock >= 2)
            sdata[tid] = sdata[tid] > sdata[tid+1] ? sdata[tid] : sdata[tid+1];
    }
    if (tid == 0)
        dev_output[blockIdx.x] = sdata[0];
}

/**
 * Struct with functions to count the norm through reduction.
*/
template< typename Traits, typename Solver>
struct Reduction
{
    using real = typename Traits::real;
    using idx = typename Traits::idx;

    /**
    * Finishes the reduction on host.
    */
    static bool host_reduction(real* dev, real& norm, idx size)
    {
        real *host;
        host = (real*)malloc(size*sizeof(real));
        if (cudaMemcpy(host, dev, size*sizeof(real),cudaMemcpyDeviceToHost) != cudaSuccess)
        {
            cudaFree(dev);
            std::cerr << "Unable to copy data from the device to the host. Error occurred while copying results in host_reduction method of the Reduction struct.\n";
            return false;
        }
        cudaFree(dev);
        for (idx i = 1; i < size; i++)
            host[0] = host[0] > host[i] ? host[0] : host[i];
        norm = host[0];
        free(host);
        return true;
    }

    /**
    * Auxiliary function for the reduction method. Counts the reduction on GPU through the kernel above and reduces the size.
    */
    static bool reduction_aux(real* dev_output, real* dev_input, idx& blocksCount, idx& size, idx i, idx small_enough)
    {
        blocksCount = size/reductionThreadsPerBlock + (size%reductionThreadsPerBlock != 0);
        global_reduction<Traits, reductionThreadsPerBlock><<<blocksCount, reductionThreadsPerBlock, reductionThreadsPerBlock*sizeof(real)>>>(dev_output, dev_input, size);
        if (!Solver::kernelErr(cudaGetLastError(), "global_reduction in reduction method of Reduction struct at iter " + std::to_string(i)))
        {
            cudaFree(dev_input);
            cudaFree(dev_output);
            return false;
        }
        cudaFree(dev_input);
        cudaDeviceSynchronize();
        size /= reductionThreadsPerBlock;
        if (size > small_enough)
        {
            blocksCount = size/reductionThreadsPerBlock + (size%reductionThreadsPerBlock != 0);
            if (cudaMalloc((void**)&dev_input, blocksCount*sizeof(real)) != cudaSuccess)
            {
                cudaFree(dev_output);
                std::cerr << "Unable to allocate on device. Error occured in global_reduction in reduction method of Reduction struct. Malloc at iter " + std::to_string(i) + "\n";
                return false;
            }
        }
        return true;
    }

    /**
    * The method for counting the norm through reduction. USes the auxiliary function, which use the kernel above.
    */
    static bool reduction(real* dev_norm, real& norm, idx size, idx max, idx small_enough)
    {
        /**
         * small_enough is how small should the array be for us to compute it on CPU. If it is too small, then it can cause errors in the kernel.
         */
        if (small_enough < 2*reductionThreadsPerBlock)
        {
            std::cerr << "Small enough must be bigger or equal than " + std::to_string(2*reductionThreadsPerBlock) + ". Error occured in global_reduction in reduction method of Reduction struct.\n";
            return false;
        }

        /**
        * Counts the number of blocks.
        */
        idx blocksCount;
        if (size < max)
            blocksCount = size/reductionThreadsPerBlock + (size%reductionThreadsPerBlock != 0);
        else
            blocksCount = max/reductionThreadsPerBlock + (max%reductionThreadsPerBlock != 0);

        /**
        * Allocates one array (dev_1) for reduction.
        */
        real *dev_1;
        if (cudaMalloc((void**)&dev_1, blocksCount*sizeof(real)) != cudaSuccess)
        {
            std::cerr << "Unable to allocate on device. Error occured in global_reduction in reduction method of Reduction struct. First malloc.\n";
            return false;
        }

        /**
        * Reduces the values for norm, which were computed in global_k5 kernel and sends the result into dev_1.
        */
        global_reduction<Traits, reductionThreadsPerBlock><<<blocksCount, reductionThreadsPerBlock, reductionThreadsPerBlock*sizeof(real)>>>(dev_1, dev_norm, size);
        if (!Solver::kernelErr(cudaGetLastError(), "global_reduction in reduction method of Reduction struct"))
        {
            cudaFree(dev_1);
            return false;
        }
        cudaDeviceSynchronize();

        /**
        * If dev_1 is already small enough, then we will proceed to compute the norm on CPU. If not then we will allocate dev_2 and use
        * the above kernel with dev_1 as an input and dev_2 as an output.
        * 
        * After that, if it is still not small enough, the dev_2 will be input and dev_1 will be output, etc.
        */
        size /= reductionThreadsPerBlock;
        if (size > small_enough)
        {
            idx i = 0;
            real *dev_2;
            blocksCount = size/reductionThreadsPerBlock + (size%reductionThreadsPerBlock != 0);
            if (cudaMalloc((void**)&dev_2, blocksCount*sizeof(real)) != cudaSuccess)
            {
                std::cerr << "Unable to allocate on device. Error occured in global_reduction in reduction method of Reduction struct. Second malloc\n";
                cudaFree(dev_1);
                return false;
            }
            while (size > small_enough)
            {
                i++;
                if (i%2 == 1)
                {
                    if (!reduction_aux(dev_2, dev_1, blocksCount, size, i, small_enough))
                        return false;
                }
                else
                {
                    if (!reduction_aux(dev_1, dev_2, blocksCount, size, i, small_enough))
                        return false;
                }
            }
            if (i%2 == 1)
                return host_reduction(dev_2, norm, size);
            else
                return host_reduction(dev_1, norm, size);

        }
        return host_reduction(dev_1, norm, size);
    }
};



#endif
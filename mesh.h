#ifndef __MESH_H__
#define __MESH_H__

#include "traits.h"
#include <algorithm>
#include <fstream>
#include <unistd.h>
#include <limits.h>

/**
 * Mesh data used in kernels.
 */
template< typename Traits>
struct Data
{
    using real = typename Traits::real;
    using idx = typename Traits::idx;
    real X, Y, hx, hy, time_step;
    idx dis_X, dis_Y;
    dim3 grids, gridIdx, blocksCount;
};

/**
 * The mesh. Solution is in dev_input.
 */
template< typename Traits>
struct Mesh
{
    using real = typename Traits::real;
    using idx = typename Traits::idx;
    using map_t = typename Traits::map_t;

    real *dev_input, *dev_u_add, *dev_norm, *dev_red_norm, *dev_k1, *dev_k2, *dev_k3, *dev_k4;
    real *host_storage, *host_norm;
    std::string directory_path;
    map_t result_num;
    idx size, size_red;
    bool failed_cuda_allocation;

    Data<Traits> data;

    Mesh(ConstructParam<Traits> param)
    {
        /**
         * Sets up data for writing solutions.
         */
        char buffer[PATH_MAX];
        if (getcwd(buffer, sizeof(buffer)) != NULL)
            directory_path = std::string(buffer) + "/results/";
        else
            std::cerr << "getcwd() error\n";


        /**
         * Sets up mesh data.
         */
        data.X = param.X;
        data.Y = param.Y;
        data.dis_X = param.dis_X;
        data.dis_Y = param.dis_Y;
        data.hx = data.X/(data.dis_X-1);
        data.hy = data.Y/(data.dis_Y-1);
        data.time_step = param.time_step;
        data.grids = param.grids;
        data.blocksCount = param.blocksCount;
        size = data.dis_X*data.dis_Y;
        size_red = param.blocksCount.x*param.blocksCount.y;
        result_num = 0;

        /**
         * Allocates data on device. The data for reduction is NOT allocated here.
         */
        if (cudaMalloc((void**)&dev_input, size*sizeof(real)) != cudaSuccess ||
            cudaMalloc((void**)&dev_u_add, size*sizeof(real)) != cudaSuccess ||
            cudaMalloc((void**)&dev_norm, size*sizeof(real)) != cudaSuccess ||
            cudaMalloc((void**)&dev_k1, size*sizeof(real)) != cudaSuccess ||
            cudaMalloc((void**)&dev_k2, size*sizeof(real)) != cudaSuccess ||
            cudaMalloc((void**)&dev_k3, size*sizeof(real)) != cudaSuccess ||
            cudaMalloc((void**)&dev_k4, size*sizeof(real)) != cudaSuccess)
        {
            failed_cuda_allocation = true;
            std::cerr << "Unable to allocate on device. Error occured in the constructor of the Mesh struct.\n";
        }
        else
        {
            failed_cuda_allocation = false;
        }

        /**
         * Allocates data on host.
         */
        host_storage = (real*)malloc(size*sizeof(real));
    }
    ~Mesh()
    {
        if (dev_input) cudaFree(dev_input);
        if (dev_u_add) cudaFree(dev_u_add);
        if (dev_norm) cudaFree(dev_norm);
        if (dev_k1) cudaFree(dev_k1);
        if (dev_k2) cudaFree(dev_k2);
        if (dev_k3) cudaFree(dev_k3);
        if (dev_k4) cudaFree(dev_k4);
        if (host_storage) free(host_storage);
    }

    /**
     * Moves solution from device to host.
     */
    bool outputToHost()
    {
        if (cudaMemcpy(host_storage, dev_input, size*sizeof(real),cudaMemcpyDeviceToHost) != cudaSuccess)
        {
            std::cerr << "Unable to copy data from the device to the host. Error occurred while copying results in outputToHost method of the Mesh struct.\n";
            return false;
        }
        return true;
    }

    /**
     * Writes the solution.
     */
    bool writeSolution()
    {
        std::ofstream file;
        std::string aux = directory_path + std::to_string(result_num) + ".txt";
        file.open(aux, std::ofstream::trunc);
        if( !file )
        {
            std::cerr << "Unable to open the file " << aux << "\n";
            return false;
        }
        result_num++;
        for(idx y = 0; y < data.dis_Y; y++)
        {
            for (idx x = 0; x < data.dis_X; x++)
            {
                file << x*data.hx << " " << y*data.hy << " " << host_storage[POS(x,y,data.dis_X,data.dis_Y)] << "\n";
            }
        }
        file.close();
        return true;
    }

};

#endif
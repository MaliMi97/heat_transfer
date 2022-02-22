#ifndef __SOLVER_H__
#define __SOLVER_H__

#include "mesh.h"
#include "kernel.h"
#include <cuda.h>
#include <cuda_runtime_api.h>

/**
 * The solver for the heat equation.
 */
template< typename Traits>
struct Solver
{
    using real = typename Traits::real;
    using idx = typename Traits::idx;

    real precision;
    real time;
    real final_time;
    dim3 threadsPerBlock, blocksPerGrid, grids, maxGridSize, blocksCount;
    Mesh<Traits> mesh;

    Solver(ConstructParam<Traits> param)
     : mesh(param)
    {
        /**
         * Sets up data such as time, final time, precision.
         * Sets up CUDA data, which will be used to count the number of blocks.
         */
        time = 0;
        final_time = param.final_time;
        precision = param.precision;
        threadsPerBlock = param.threadsPerBlock;
        grids = param.grids;
        maxGridSize = param.maxGridSize;
        blocksCount = param.blocksCount;
    }
    
    /**
     * Device function for setting up the Dirichlet boundary.
     */
    __device__ static real dirichletBoundary() { return 0;}

    /**
     * Device function for setting up the initial condition.
     */
    __device__ static real initialCondition(real x, real y, real xs, real ys, real R) { return ((x - xs)*(x - xs) + (y - ys)*(y - ys) -  R*R > 0) ? 1.0 : 0.0;}
    
    /**
     * Device functions for computing the right-hand side.
     */
    __device__ static real rightHandSide(real u_zz, real u_pz, real u_zp, real u_mz, real u_zm, real hx, real hy)
    { 
        return (u_mz - 2*u_zz + u_pz)/(hx*hx) + (u_zm - 2*u_zz + u_zp)/(hy*hy);
    }
    __device__ static real rightHandSideBoundary(real u_zz, real u_1, real u_2, real h)
    { 
        return (u_zz - 2*u_1 + u_2)/(h*h);
    }

    /**
     * Computes how the number of blocks per grid.
     */
    void computeBlocksPerGrid(idx gridIdx_x, idx gridIdx_y)
    {
        if (gridIdx_x < blocksCount.x/maxGridSize.x)
                blocksPerGrid.x = maxGridSize.x;
        else
            blocksPerGrid.x = blocksCount.x%maxGridSize.x;

        if (gridIdx_y < blocksCount.y/maxGridSize.y)
            blocksPerGrid.y = maxGridSize.y;
        else
            blocksPerGrid.y = blocksCount.y%maxGridSize.y;
    }

    /**
     * Function for writing an error when kernel fails.
     */
    static bool kernelErr(cudaError err, std::string kernel)
    {
        if (err != cudaSuccess)
        {
            std::cerr << "Computation on the device failed with error: " << cudaGetErrorString(err) << ". Error occurred at kernel " << kernel << "\n";
            return false;
        }
        return true;
    }

    /**
     * Computes the time step.
     */
    void computeTimeStep(real norm)
    {
        real aux = 0.8*mesh.data.time_step*std::pow(precision/norm, 0.2);
        mesh.data.time_step = aux < final_time - time ? aux : final_time - time; 
    }

};

#endif
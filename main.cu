#include <iostream>
#include <fstream>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <chrono>
#include "solver.h"
#include "kernel.h"
#include "reduction.h"

/**
 * The problem is being initialized and solved in this function.
 */
template<typename Traits, typename Data, typename Solver, typename KernelDeviceMesh, typename Reduction>
bool solve(Solver& solver)
{
    using real = typename Traits::real;
    using idx = typename Traits::idx;

    /**
     * The current iteration. 0 is for the initial condition.
     */
    idx iter = 0;

    /**
     * Sets up the initial and boundary conditions.
     */
    for ( solver.mesh.data.gridIdx.x = 0; solver.mesh.data.gridIdx.x < solver.grids.x; solver.mesh.data.gridIdx.x++)
        for ( solver.mesh.data.gridIdx.y = 0; solver.mesh.data.gridIdx.y < solver.grids.y; solver.mesh.data.gridIdx.y++)
        {
            solver.computeBlocksPerGrid(solver.mesh.data.gridIdx.x, solver.mesh.data.gridIdx.y);
            global_init<Traits, Data, Solver><<<solver.blocksPerGrid,solver.threadsPerBlock>>>(
                solver.mesh.dev_input, 
                solver.mesh.dev_k1,
                solver.mesh.dev_k2,
                solver.mesh.dev_k3,
                solver.mesh.dev_k4,
                solver.mesh.data);
            if (!Solver::kernelErr(cudaGetLastError(), "global_init"))
                return false;
        }
    solver.mesh.outputToHost();
    solver.mesh.writeSolution();

    /**
     * Solves the problem.
     */
    while (solver.time < solver.final_time)
    {
        /**
         * Computes k1 on the whole computation area except the boundary.
         */
        for ( solver.mesh.data.gridIdx.x = 0; solver.mesh.data.gridIdx.x < solver.grids.x; solver.mesh.data.gridIdx.x++)
            for ( solver.mesh.data.gridIdx.y = 0; solver.mesh.data.gridIdx.y < solver.grids.y; solver.mesh.data.gridIdx.y++)
            {
                solver.computeBlocksPerGrid(solver.mesh.data.gridIdx.x, solver.mesh.data.gridIdx.y);
                global_k1<Traits, Data, Solver, KernelDeviceMesh><<<solver.blocksPerGrid,solver.threadsPerBlock>>>(solver.mesh.dev_input, solver.mesh.dev_k1, solver.mesh.data);
                if (!Solver::kernelErr(cudaGetLastError(), "global_k1"))
                    return false;
            }
        cudaDeviceSynchronize();

        /**
         * Computes k2 on the whole computation area except the boundary.
         */
        for ( solver.mesh.data.gridIdx.x = 0; solver.mesh.data.gridIdx.x < solver.grids.x; solver.mesh.data.gridIdx.x++)
            for ( solver.mesh.data.gridIdx.y = 0; solver.mesh.data.gridIdx.y < solver.grids.y; solver.mesh.data.gridIdx.y++)
            {
                solver.computeBlocksPerGrid(solver.mesh.data.gridIdx.x, solver.mesh.data.gridIdx.y);
                global_k2<Traits, Data, Solver, KernelDeviceMesh><<<solver.blocksPerGrid,solver.threadsPerBlock>>>(solver.mesh.dev_input, solver.mesh.dev_k1, solver.mesh.dev_k2, solver.mesh.data);
                if (!Solver::kernelErr(cudaGetLastError(), "global_k2"))
                    return false;
            }
        cudaDeviceSynchronize();

        /**
         * Computes k3 on the whole computation area except the boundary.
         */
        for ( solver.mesh.data.gridIdx.x = 0; solver.mesh.data.gridIdx.x < solver.grids.x; solver.mesh.data.gridIdx.x++)
            for ( solver.mesh.data.gridIdx.y = 0; solver.mesh.data.gridIdx.y < solver.grids.y; solver.mesh.data.gridIdx.y++)
            {
                solver.computeBlocksPerGrid(solver.mesh.data.gridIdx.x, solver.mesh.data.gridIdx.y);
                global_k3<Traits, Data, Solver, KernelDeviceMesh><<<solver.blocksPerGrid,solver.threadsPerBlock>>>(solver.mesh.dev_input, solver.mesh.dev_k1, solver.mesh.dev_k2, solver.mesh.dev_k3, solver.mesh.data);
                if (!Solver::kernelErr(cudaGetLastError(), "global_k3"))
                    return false;
            }
        cudaDeviceSynchronize();

        /**
         * Computes k4 on the whole computation area except the boundary.
         */
        for ( solver.mesh.data.gridIdx.x = 0; solver.mesh.data.gridIdx.x < solver.grids.x; solver.mesh.data.gridIdx.x++)
            for ( solver.mesh.data.gridIdx.y = 0; solver.mesh.data.gridIdx.y < solver.grids.y; solver.mesh.data.gridIdx.y++)
            {
                solver.computeBlocksPerGrid(solver.mesh.data.gridIdx.x, solver.mesh.data.gridIdx.y);
                global_k4<Traits, Data, Solver, KernelDeviceMesh><<<solver.blocksPerGrid,solver.threadsPerBlock>>>(solver.mesh.dev_input, solver.mesh.dev_k1, solver.mesh.dev_k3, solver.mesh.dev_k4, solver.mesh.data);
                if (!Solver::kernelErr(cudaGetLastError(), "global_k4"))
                    return false;
            }
        cudaDeviceSynchronize();

        /**
         * Computes k5 and then computes the value, which will be used to count the norm in each point except the boundary.
         * Also computes the value, which will be added to the dev_input if the norm is small enough.
         */
        for ( solver.mesh.data.gridIdx.x = 0; solver.mesh.data.gridIdx.x < solver.grids.x; solver.mesh.data.gridIdx.x++)
            for ( solver.mesh.data.gridIdx.y = 0; solver.mesh.data.gridIdx.y < solver.grids.y; solver.mesh.data.gridIdx.y++)
            {
                solver.computeBlocksPerGrid(solver.mesh.data.gridIdx.x, solver.mesh.data.gridIdx.y);
                global_k5<Traits, Data, Solver, KernelDeviceMesh><<<solver.blocksPerGrid,solver.threadsPerBlock>>>(solver.mesh.dev_input, solver.mesh.dev_k1, solver.mesh.dev_k3, solver.mesh.dev_k4, solver.mesh.dev_u_add, solver.mesh.dev_norm, solver.mesh.data);
                if (!Solver::kernelErr(cudaGetLastError(), "global_k5"))
                    return false;
            }
        cudaDeviceSynchronize();
        
        /**
         * Computes the norm.
         */
        real norm;
        if (!Reduction::reduction(solver.mesh.dev_norm, norm, solver.mesh.size, solver.maxGridSize.x, 512))
            return false;
        norm /= 3.0;

        /**
         * Modifies dev_input if norm is small enough. Also sometimes writes the solution.
         */
        if (norm < solver.precision)
        {
            /**
             * Modifies dev_input.
             */
            for ( solver.mesh.data.gridIdx.x = 0; solver.mesh.data.gridIdx.x < solver.grids.x; solver.mesh.data.gridIdx.x++)
                for ( solver.mesh.data.gridIdx.y = 0; solver.mesh.data.gridIdx.y < solver.grids.y; solver.mesh.data.gridIdx.y++)
                {
                    solver.computeBlocksPerGrid(solver.mesh.data.gridIdx.x, solver.mesh.data.gridIdx.y);
                    global_add<Traits, Data><<<solver.blocksPerGrid,solver.threadsPerBlock>>>(solver.mesh.dev_input, solver.mesh.dev_u_add, solver.mesh.data);
                    if (!Solver::kernelErr(cudaGetLastError(), "global_add"))
                        return false;
                }
            cudaDeviceSynchronize();

            /**
             * Increases time.
             */
            solver.time += solver.mesh.data.time_step;

            /**
             * Writes solution.
             */
            if (iter%1000 == 0 || solver.time >= solver.final_time)
            {
                if (!solver.mesh.outputToHost())
                    return false;
                if (!solver.mesh.writeSolution())
                    return false;
                printf("%li: %e out of %e with %e\n", iter, solver.time, solver.final_time, solver.mesh.data.time_step);
            }
            iter++;
        }
        /**
         * Computes new time step.
         */
        solver.computeTimeStep(norm);
    }
    return true;
}

int main() 
{
    /**
     * Starts measuring the runtime.
     */
    auto start = std::chrono::high_resolution_clock::now();

    /**
     * Sets up parameters.
     */
    ConstructParam<TraitsDP> param;
    param.X = 1;    //the physical length
    param.Y = 1;    //the physical height
    param.dis_X = 128;  //the number of points, which discretize X
    param.dis_Y = 128;  //the number of points, which discretize Y
    param.time_step = 1e-4; //the initial time step
    param.final_time = 0.5; //the computation ends when the final time is reached
    param.precision = 1e-8; //how precise do we want each iteration to be 

    /**
     * Sets up CUDA parameters.
     */
    cudaDeviceProp properties;
    cudaGetDeviceProperties(&properties, 0);
    param.maxGridSize = {(unsigned int)properties.maxGridSize[0], (unsigned int)properties.maxGridSize[1]};
    param.threadsPerBlock = {16,16};
    param.blocksCount = {(unsigned int)(param.dis_X/param.threadsPerBlock.x + (param.dis_X%param.threadsPerBlock.x != 0)),
        (unsigned int)(param.dis_Y/param.threadsPerBlock.y + (param.dis_Y%param.threadsPerBlock.y != 0))};
    param.grids = {param.blocksCount.x/param.maxGridSize.x + (param.blocksCount.x%param.maxGridSize.x != 0),
        param.blocksCount.y/param.maxGridSize.y + (param.blocksCount.y%param.maxGridSize.y != 0)};

    /**
     * Sets up mesh and equation.
     */
    Solver<TraitsDP> solver(param);

    /**
     * Solves the equation.
     */
    if(solver.mesh.failed_cuda_allocation || !solve<TraitsDP, Data<TraitsDP>, Solver<TraitsDP>, KernelDeviceMesh<TraitsDP>, Reduction<TraitsDP, Solver<TraitsDP>>>(solver))
    {
        return EXIT_FAILURE;
    }

    /**
     * Stops measuring runtime.
     */
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

    /**
     * Writes runtime.
     */
    std::ofstream file;
    file.open(solver.mesh.directory_path + "time.txt", std::ofstream::trunc);
    if( !file )
    {
        std::cerr << "Unable to open the file " << solver.mesh.directory_path + "time.txt" << "\n";
        return false;
    }
    file << duration.count();
    file.close();
    return 0;
}

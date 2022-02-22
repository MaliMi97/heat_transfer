#ifndef __KERNEL_H__
#define __KERNEL_H__

#include "traits.h"

/**
 * Struct used in kernels for computing k1 - k5.
**/
template< typename real>
struct KernelStruct
{
    real u_zz, u_pz, u_zp, u_mz, u_zm, u_add, norm, u_1, u_2;
};

/**
 * Struct, which contains device function for getting input data.
 */
template< typename Traits>
struct KernelDeviceMesh
{
    using real = typename Traits::real;
    using idx = typename Traits::idx;

    /**
    * Gets input data for k1.
    */
    __device__ static void k1_input(KernelStruct<real>& KS, real* dev_input, idx x, idx y, idx X, idx Y)
    {
        KS.u_zz = dev_input[POS(x  ,y  ,X,Y)];
        KS.u_pz = dev_input[POS(x+1,y  ,X,Y)];
        KS.u_zp = dev_input[POS(x  ,y+1,X,Y)];
        KS.u_mz = dev_input[POS(x-1,y  ,X,Y)];
        KS.u_zm = dev_input[POS(x  ,y-1,X,Y)];
    }
    __device__ static void k1_inputBoundary(KernelStruct<real>& KS, real* dev_input, idx x, idx y, idx X, idx Y, int x1, int x2, int y1, int y2)
    {
        KS.u_zz = dev_input[POS(x   ,y   ,X,Y)];
        KS.u_1  = dev_input[POS(x+x1,y+y1,X,Y)];
        KS.u_2  = dev_input[POS(x+x2,y+y2,X,Y)];
    }

    /**
    * Gets modified input data for k2.
    */
    __device__ static void k2_input(KernelStruct<real>& KS, real* dev_input, real* dev_k1, idx x, idx y, idx X, idx Y)
    {
        KS.u_zz = dev_input[POS(x  ,y  ,X,Y)] + dev_k1[POS(x  ,y  ,X,Y)]/3;
        KS.u_pz = dev_input[POS(x+1,y  ,X,Y)] + dev_k1[POS(x+1,y  ,X,Y)]/3;
        KS.u_zp = dev_input[POS(x  ,y+1,X,Y)] + dev_k1[POS(x  ,y+1,X,Y)]/3;
        KS.u_mz = dev_input[POS(x-1,y  ,X,Y)] + dev_k1[POS(x-1,y  ,X,Y)]/3;
        KS.u_zm = dev_input[POS(x  ,y-1,X,Y)] + dev_k1[POS(x  ,y-1,X,Y)]/3;
    }
    __device__ static void k2_inputBoundary(KernelStruct<real>& KS, real* dev_input, real* dev_k1, idx x, idx y, idx X, idx Y, int x1, int x2, int y1, int y2)
    {
        KS.u_zz = dev_input[POS(x   ,y   ,X,Y)] + dev_k1[POS(x   ,y   ,X,Y)]/3;
        KS.u_1  = dev_input[POS(x+x1,y+y1,X,Y)] + dev_k1[POS(x+x1,y+y1,X,Y)]/3;
        KS.u_2  = dev_input[POS(x+x2,y+y2,X,Y)] + dev_k1[POS(x+x1,y+y2,X,Y)]/3;
    }

    /**
    * Gets modified input data for k3.
    */
    __device__ static void k3_input(KernelStruct<real>& KS, real* dev_input, real* dev_k1, real* dev_k2, idx x, idx y, idx X, idx Y)
    {
        KS.u_zz = dev_input[POS(x  ,y  ,X,Y)] + (dev_k1[POS(x  ,y  ,X,Y)] + dev_k2[POS(x  ,y  ,X,Y)])/6;
        KS.u_pz = dev_input[POS(x+1,y  ,X,Y)] + (dev_k1[POS(x+1,y  ,X,Y)] + dev_k2[POS(x+1,y  ,X,Y)])/6;
        KS.u_zp = dev_input[POS(x  ,y+1,X,Y)] + (dev_k1[POS(x  ,y+1,X,Y)] + dev_k2[POS(x  ,y+1,X,Y)])/6;
        KS.u_mz = dev_input[POS(x-1,y  ,X,Y)] + (dev_k1[POS(x-1,y  ,X,Y)] + dev_k2[POS(x-1,y  ,X,Y)])/6;
        KS.u_zm = dev_input[POS(x  ,y-1,X,Y)] + (dev_k1[POS(x  ,y-1,X,Y)] + dev_k2[POS(x  ,y-1,X,Y)])/6;
    }
    __device__ static void k3_inputBoundary(KernelStruct<real>& KS, real* dev_input, real* dev_k1, real* dev_k2, idx x, idx y, idx X, idx Y, int x1, int x2, int y1, int y2)
    {
        KS.u_zz = dev_input[POS(x   ,y   ,X,Y)] + (dev_k1[POS(x   ,y   ,X,Y)] + dev_k2[POS(x   ,y   ,X,Y)])/6;
        KS.u_1  = dev_input[POS(x+x1,y+y1,X,Y)] + (dev_k1[POS(x+x1,y+y1,X,Y)] + dev_k2[POS(x+x1,y+y1,X,Y)])/6;
        KS.u_2  = dev_input[POS(x+x2,y+y2,X,Y)] + (dev_k1[POS(x+x2,y+y2,X,Y)] + dev_k2[POS(x+x2,y+y2,X,Y)])/6;
    }

    /**
    * Gets modified input data for k4.
    */
    __device__ static void k4_input(KernelStruct<real>& KS, real* dev_input, real* dev_k1, real* dev_k3, idx x, idx y, idx X, idx Y)
    {
        KS.u_zz = dev_input[POS(x  ,y  ,X,Y)] + 0.125*dev_k1[POS(x  ,y  ,X,Y)] + 0.375*dev_k3[POS(x  ,y  ,X,Y)];
        KS.u_pz = dev_input[POS(x+1,y  ,X,Y)] + 0.125*dev_k1[POS(x+1,y  ,X,Y)] + 0.375*dev_k3[POS(x+1,y  ,X,Y)];
        KS.u_zp = dev_input[POS(x  ,y+1,X,Y)] + 0.125*dev_k1[POS(x  ,y+1,X,Y)] + 0.375*dev_k3[POS(x  ,y+1,X,Y)];
        KS.u_mz = dev_input[POS(x-1,y  ,X,Y)] + 0.125*dev_k1[POS(x-1,y  ,X,Y)] + 0.375*dev_k3[POS(x-1,y  ,X,Y)];
        KS.u_zm = dev_input[POS(x  ,y-1,X,Y)] + 0.125*dev_k1[POS(x  ,y-1,X,Y)] + 0.375*dev_k3[POS(x  ,y-1,X,Y)];
    }
    __device__ static void k4_inputBoundary(KernelStruct<real>& KS, real* dev_input, real* dev_k1, real* dev_k3, idx x, idx y, idx X, idx Y, int x1, int x2, int y1, int y2)
    {
        KS.u_zz = dev_input[POS(x   ,y   ,X,Y)] + 0.125*dev_k1[POS(x   ,y   ,X,Y)] + 0.375*dev_k3[POS(x   ,y   ,X,Y)];
        KS.u_1  = dev_input[POS(x+x1,y+y1,X,Y)] + 0.125*dev_k1[POS(x+x1,y+y1,X,Y)] + 0.375*dev_k3[POS(x+x1,y+y1,X,Y)];
        KS.u_2  = dev_input[POS(x+x2,y+y2,X,Y)] + 0.125*dev_k1[POS(x+x2,y+y2,X,Y)] + 0.375*dev_k3[POS(x+x2,y+y2,X,Y)];
    }

    /**
    * Gets modified input data for k5.
    */
    __device__ static void k5_input(KernelStruct<real>& KS, real* dev_input, real* dev_k1, real* dev_k3, real* dev_k4, idx x, idx y, idx X, idx Y)
    {
        KS.u_zz = dev_input[POS(x  ,y  ,X,Y)] + 0.5*dev_k1[POS(x  ,y  ,X,Y)] - 1.5*dev_k3[POS(x  ,y  ,X,Y)] + 2.0*dev_k4[POS(x  ,y  ,X,Y)];
        KS.u_pz = dev_input[POS(x+1,y  ,X,Y)] + 0.5*dev_k1[POS(x+1,y  ,X,Y)] - 1.5*dev_k3[POS(x+1,y  ,X,Y)] + 2.0*dev_k4[POS(x+1,y  ,X,Y)];
        KS.u_zp = dev_input[POS(x  ,y+1,X,Y)] + 0.5*dev_k1[POS(x  ,y+1,X,Y)] - 1.5*dev_k3[POS(x  ,y+1,X,Y)] + 2.0*dev_k4[POS(x  ,y+1,X,Y)];
        KS.u_mz = dev_input[POS(x-1,y  ,X,Y)] + 0.5*dev_k1[POS(x-1,y  ,X,Y)] - 1.5*dev_k3[POS(x-1,y  ,X,Y)] + 2.0*dev_k4[POS(x-1,y  ,X,Y)];
        KS.u_zm = dev_input[POS(x  ,y-1,X,Y)] + 0.5*dev_k1[POS(x  ,y-1,X,Y)] - 1.5*dev_k3[POS(x  ,y-1,X,Y)] + 2.0*dev_k4[POS(x  ,y-1,X,Y)];
    }
    __device__ static void k5_inputBoundary(KernelStruct<real>& KS, real* dev_input, real* dev_k1, real* dev_k3, real* dev_k4, idx x, idx y, idx X, idx Y, int x1, int x2, int y1, int y2)
    {
        KS.u_zz = dev_input[POS(x   ,y   ,X,Y)] + 0.5*dev_k1[POS(x   ,y   ,X,Y)] - 1.5*dev_k3[POS(x   ,y   ,X,Y)] + 2.0*dev_k4[POS(x   ,y   ,X,Y)];
        KS.u_1  = dev_input[POS(x+x1,y+y1,X,Y)] + 0.5*dev_k1[POS(x+x1,y+y1,X,Y)] - 1.5*dev_k3[POS(x+x1,y+y1,X,Y)] + 2.0*dev_k4[POS(x+x1,y+y1,X,Y)];
        KS.u_2  = dev_input[POS(x+x2,y+y2,X,Y)] + 0.5*dev_k1[POS(x+x2,y+y2,X,Y)] - 1.5*dev_k3[POS(x+x2,y+y2,X,Y)] + 2.0*dev_k4[POS(x+x2,y+y2,X,Y)];
    }

};

/**
 * Kernel for computing the k1.
 */
template<typename Traits, typename Data, typename Solver, typename KernelDeviceMesh>
__global__ void global_k1(typename Traits::real* dev_input, typename Traits::real* dev_k1, Data data)
{
    using idx = typename Traits::idx;
    using real = typename Traits::real;

    idx x = threadIdx.x + (blockIdx.x + data.gridIdx.x*data.grids.x)*blockDim.x;
	idx y = threadIdx.y + (blockIdx.y + data.gridIdx.y*data.grids.y)*blockDim.y;

    KernelStruct<real> KS;

    if (x == 0)
    {
        KernelDeviceMesh::k1_inputBoundary(KS, dev_input, x, y, data.dis_X, data.dis_Y, 1, 2, 0, 0);
        dev_k1[POS(x,y,data.dis_X,data.dis_Y)] = data.time_step*Solver::rightHandSideBoundary(KS.u_zz, KS.u_1, KS.u_2, data.hx);
    }
    else if (x == data.dis_X-1)
    {
        KernelDeviceMesh::k1_inputBoundary(KS, dev_input, x, y, data.dis_X, data.dis_Y, -1, -2, 0, 0);
        dev_k1[POS(x,y,data.dis_X,data.dis_Y)] = data.time_step*Solver::rightHandSideBoundary(KS.u_zz, KS.u_1, KS.u_2, data.hx);
    }
    else if (y == 0)
    {
        KernelDeviceMesh::k1_inputBoundary(KS, dev_input, x, y, data.dis_X, data.dis_Y, 0, 0, 1, 2);
        dev_k1[POS(x,y,data.dis_X,data.dis_Y)] = data.time_step*Solver::rightHandSideBoundary(KS.u_zz, KS.u_1, KS.u_2, data.hy);
    }
    else if (y == data.dis_Y-1)
    {
        KernelDeviceMesh::k1_inputBoundary(KS, dev_input, x, y, data.dis_X, data.dis_Y, 0, 0, -1, -2);
        dev_k1[POS(x,y,data.dis_X,data.dis_Y)] = data.time_step*Solver::rightHandSideBoundary(KS.u_zz, KS.u_1, KS.u_2, data.hy);
    }
    else
    {
        KernelDeviceMesh::k1_input(KS, dev_input, x, y, data.dis_X, data.dis_Y);
        dev_k1[POS(x,y,data.dis_X,data.dis_Y)] = data.time_step*Solver::rightHandSide(KS.u_zz, KS.u_pz, KS.u_zp, KS.u_mz, KS.u_zm, data.hx, data.hy);
    }
}

/**
 * Kernel for computing the k2.
 */
template<typename Traits, typename Data, typename Solver, typename KernelDeviceMesh>
__global__ void global_k2(typename Traits::real* dev_input, typename Traits::real* dev_k1, typename Traits::real* dev_k2, Data data)
{
    using idx = typename Traits::idx;
    using real = typename Traits::real;

    idx x = threadIdx.x + (blockIdx.x + data.gridIdx.x*data.grids.x)*blockDim.x;
	idx y = threadIdx.y + (blockIdx.y + data.gridIdx.y*data.grids.y)*blockDim.y;

    KernelStruct<real> KS;

    if (x == 0)
    {
        KernelDeviceMesh::k2_inputBoundary(KS, dev_input, dev_k1, x, y, data.dis_X, data.dis_Y, 1, 2, 0, 0);
        dev_k2[POS(x,y,data.dis_X,data.dis_Y)] = data.time_step*Solver::rightHandSideBoundary(KS.u_zz, KS.u_1, KS.u_2, data.hx);
    }
    else if (x == data.dis_X-1)
    {
        KernelDeviceMesh::k2_inputBoundary(KS, dev_input, dev_k1, x, y, data.dis_X, data.dis_Y, -1, -2, 0, 0);
        dev_k2[POS(x,y,data.dis_X,data.dis_Y)] = data.time_step*Solver::rightHandSideBoundary(KS.u_zz, KS.u_1, KS.u_2, data.hx);
    }
    else if (y == 0)
    {
        KernelDeviceMesh::k2_inputBoundary(KS, dev_input, dev_k1, x, y, data.dis_X, data.dis_Y, 0, 0, 1, 2);
        dev_k2[POS(x,y,data.dis_X,data.dis_Y)] = data.time_step*Solver::rightHandSideBoundary(KS.u_zz, KS.u_1, KS.u_2, data.hy);
    }
    else if (y == data.dis_Y-1)
    {
        KernelDeviceMesh::k2_inputBoundary(KS, dev_input, dev_k1, x, y, data.dis_X, data.dis_Y, 0, 0, -1, -2);
        dev_k2[POS(x,y,data.dis_X,data.dis_Y)] = data.time_step*Solver::rightHandSideBoundary(KS.u_zz, KS.u_1, KS.u_2, data.hy);
    }
    else
    {
        KernelDeviceMesh::k2_input(KS, dev_input, dev_k1, x, y, data.dis_X, data.dis_Y);
        dev_k2[POS(x,y,data.dis_X,data.dis_Y)] = data.time_step*Solver::rightHandSide(KS.u_zz, KS.u_pz, KS.u_zp, KS.u_mz, KS.u_zm, data.hx, data.hy);
    }
}

/**
 * Kernel for computing the k3.
 */
template<typename Traits, typename Data, typename Solver, typename KernelDeviceMesh>
__global__ void global_k3(typename Traits::real* dev_input, typename Traits::real* dev_k1, typename Traits::real* dev_k2, typename Traits::real* dev_k3, Data data)
{
    using idx = typename Traits::idx;
    using real = typename Traits::real;

    idx x = threadIdx.x + (blockIdx.x + data.gridIdx.x*data.grids.x)*blockDim.x;
	idx y = threadIdx.y + (blockIdx.y + data.gridIdx.y*data.grids.y)*blockDim.y;

    KernelStruct<real> KS;

    if (x == 0)
    {
        KernelDeviceMesh::k3_inputBoundary(KS, dev_input, dev_k1, dev_k2, x, y, data.dis_X, data.dis_Y, 1, 2, 0, 0);
        dev_k3[POS(x,y,data.dis_X,data.dis_Y)] = data.time_step*Solver::rightHandSideBoundary(KS.u_zz, KS.u_1, KS.u_2, data.hx);
    }
    else if (x == data.dis_X-1)
    {
        KernelDeviceMesh::k3_inputBoundary(KS, dev_input, dev_k1, dev_k2, x, y, data.dis_X, data.dis_Y, -1, -2, 0, 0);
        dev_k3[POS(x,y,data.dis_X,data.dis_Y)] = data.time_step*Solver::rightHandSideBoundary(KS.u_zz, KS.u_1, KS.u_2, data.hx);
    }
    else if (y == 0)
    {
        KernelDeviceMesh::k3_inputBoundary(KS, dev_input, dev_k1, dev_k2, x, y, data.dis_X, data.dis_Y, 0, 0, 1, 2);
        dev_k3[POS(x,y,data.dis_X,data.dis_Y)] = data.time_step*Solver::rightHandSideBoundary(KS.u_zz, KS.u_1, KS.u_2, data.hy);
    }
    else if (y == data.dis_Y-1)
    {
        KernelDeviceMesh::k3_inputBoundary(KS, dev_input, dev_k1, dev_k2, x, y, data.dis_X, data.dis_Y, 0, 0, -1, -2);
        dev_k3[POS(x,y,data.dis_X,data.dis_Y)] = data.time_step*Solver::rightHandSideBoundary(KS.u_zz, KS.u_1, KS.u_2, data.hy);
    }
    else
    {
        KernelDeviceMesh::k3_input(KS, dev_input, dev_k1, dev_k2, x, y, data.dis_X, data.dis_Y);
        dev_k3[POS(x,y,data.dis_X,data.dis_Y)] = data.time_step*Solver::rightHandSide(KS.u_zz, KS.u_pz, KS.u_zp, KS.u_mz, KS.u_zm, data.hx, data.hy);
    }
}

/**
 * Kernel for computing the k4.
 */
template<typename Traits, typename Data, typename Solver, typename KernelDeviceMesh>
__global__ void global_k4(typename Traits::real* dev_input, typename Traits::real* dev_k1, typename Traits::real* dev_k3, typename Traits::real* dev_k4, Data data)
{
    using idx = typename Traits::idx;
    using real = typename Traits::real;

    idx x = threadIdx.x + (blockIdx.x + data.gridIdx.x*data.grids.x)*blockDim.x;
	idx y = threadIdx.y + (blockIdx.y + data.gridIdx.y*data.grids.y)*blockDim.y;

    KernelStruct<real> KS;

    if (x == 0)
    {
        KernelDeviceMesh::k4_inputBoundary(KS, dev_input, dev_k1, dev_k3, x, y, data.dis_X, data.dis_Y, 1, 2, 0, 0);
        dev_k4[POS(x,y,data.dis_X,data.dis_Y)] = data.time_step*Solver::rightHandSideBoundary(KS.u_zz, KS.u_1, KS.u_2, data.hx);
    }
    else if (x == data.dis_X-1)
    {
        KernelDeviceMesh::k4_inputBoundary(KS, dev_input, dev_k1, dev_k3, x, y, data.dis_X, data.dis_Y, -1, -2, 0, 0);
        dev_k4[POS(x,y,data.dis_X,data.dis_Y)] = data.time_step*Solver::rightHandSideBoundary(KS.u_zz, KS.u_1, KS.u_2, data.hx);
    }
    else if (y == 0)
    {
        KernelDeviceMesh::k4_inputBoundary(KS, dev_input, dev_k1, dev_k3, x, y, data.dis_X, data.dis_Y, 0, 0, 1, 2);
        dev_k4[POS(x,y,data.dis_X,data.dis_Y)] = data.time_step*Solver::rightHandSideBoundary(KS.u_zz, KS.u_1, KS.u_2, data.hy);
    }
    else if (y == data.dis_Y-1)
    {
        KernelDeviceMesh::k4_inputBoundary(KS, dev_input, dev_k1, dev_k3, x, y, data.dis_X, data.dis_Y, 0, 0, -1, -2);
        dev_k4[POS(x,y,data.dis_X,data.dis_Y)] = data.time_step*Solver::rightHandSideBoundary(KS.u_zz, KS.u_1, KS.u_2, data.hy);
    }
    else
    {
        KernelDeviceMesh::k4_input(KS, dev_input, dev_k1, dev_k3, x, y, data.dis_X, data.dis_Y);
        dev_k4[POS(x,y,data.dis_X,data.dis_Y)] = data.time_step*Solver::rightHandSide(KS.u_zz, KS.u_pz, KS.u_zp, KS.u_mz, KS.u_zm, data.hx, data.hy);
    }
}

/**
 * Kernel for computing k5, the value which will be sued to count norm and the value, which will be used to modify the input data.
 */
template<typename Traits, typename Data, typename Solver, typename KernelDeviceMesh>
__global__ void global_k5(typename Traits::real* dev_input, typename Traits::real* dev_k1, typename Traits::real* dev_k3, typename Traits::real* dev_k4, typename Traits::real* dev_u_add, typename Traits::real* dev_norm, Data data)
{
    using idx = typename Traits::idx;
    using real = typename Traits::real;

    idx x = threadIdx.x + (blockIdx.x + data.gridIdx.x*data.grids.x)*blockDim.x;
	idx y = threadIdx.y + (blockIdx.y + data.gridIdx.y*data.grids.y)*blockDim.y;

    KernelStruct<real> KS;
    real k5;

    if (x == 0)
    {
        KernelDeviceMesh::k5_inputBoundary(KS, dev_input, dev_k1, dev_k3, dev_k4, x, y, data.dis_X, data.dis_Y, 1, 2, 0, 0);
        k5 = data.time_step*Solver::rightHandSideBoundary(KS.u_zz, KS.u_1, KS.u_2, data.hx);
        dev_u_add[POS(x,y,data.dis_X,data.dis_Y)] = 0;
    }
    else if (x == data.dis_X-1)
    {
        KernelDeviceMesh::k5_inputBoundary(KS, dev_input, dev_k1, dev_k3, dev_k4, x, y, data.dis_X, data.dis_Y, -1, -2, 0, 0);
        k5 = data.time_step*Solver::rightHandSideBoundary(KS.u_zz, KS.u_1, KS.u_2, data.hx);
        dev_u_add[POS(x,y,data.dis_X,data.dis_Y)] = 0;
    }
    else if (y == 0)
    {
        KernelDeviceMesh::k5_inputBoundary(KS, dev_input, dev_k1, dev_k3, dev_k4, x, y, data.dis_X, data.dis_Y, 0, 0, 1, 2);
        k5 = data.time_step*Solver::rightHandSideBoundary(KS.u_zz, KS.u_1, KS.u_2, data.hy);
        dev_u_add[POS(x,y,data.dis_X,data.dis_Y)] = 0;
    }
    else if (y == data.dis_Y-1)
    {
        KernelDeviceMesh::k5_inputBoundary(KS, dev_input, dev_k1, dev_k3, dev_k4, x, y, data.dis_X, data.dis_Y, 0, 0, -1, -2);
        k5 = data.time_step*Solver::rightHandSideBoundary(KS.u_zz, KS.u_1, KS.u_2, data.hy);
        dev_u_add[POS(x,y,data.dis_X,data.dis_Y)] = 0;
    }
    else
    {
        KernelDeviceMesh::k5_input(KS, dev_input, dev_k1, dev_k3, dev_k4, x, y, data.dis_X, data.dis_Y);
        k5 = data.time_step*Solver::rightHandSide(KS.u_zz, KS.u_pz, KS.u_zp, KS.u_mz, KS.u_zm, data.hx, data.hy);
        dev_u_add[POS(x,y,data.dis_X,data.dis_Y)] = (dev_k1[POS(x,y,data.dis_X,data.dis_Y)] + 4.0*dev_k4[POS(x,y,data.dis_X,data.dis_Y)] + k5)/6.0;
    }

    dev_norm[POS(x,y,data.dis_X,data.dis_Y)] = (0.2*dev_k1[POS(x,y,data.dis_X,data.dis_Y)] - 0.9*dev_k3[POS(x,y,data.dis_X,data.dis_Y)] + 0.8*dev_k4[POS(x,y,data.dis_X,data.dis_Y)] - 0.1*k5)/3.0;
    if (dev_norm[POS(x,y,data.dis_X,data.dis_Y)] < 0)
        dev_norm[POS(x,y,data.dis_X,data.dis_Y)] *= -1;
}

/**
 * Kernel for setting up the initial and boundary conditions.
 */
template<typename Traits, typename Data, typename Solver>
__global__ void global_init(typename Traits::real* dev_input,typename Traits::real* dev_k1,typename Traits::real* dev_k2,typename Traits::real* dev_k3,typename Traits::real* dev_k4, Data data )
{
    using idx = typename Traits::idx;
    idx x = threadIdx.x + (blockIdx.x + data.gridIdx.x*data.grids.x)*blockDim.x;
	idx y = threadIdx.y + (blockIdx.y + data.gridIdx.y*data.grids.y)*blockDim.y;
    dev_input[POS(x, y, data.dis_X, data.dis_Y)] = Solver::initialCondition(
        x*data.hx, y*data.hy, 0.5, 0.5, 0.25); 
    if (x == 0 || x == data.dis_X-1 || y == 0 || y == data.dis_Y-1)
    {
        dev_input[POS(x, y, data.dis_X, data.dis_Y)] = Solver::dirichletBoundary();
        dev_k1[POS(x, y, data.dis_X, data.dis_Y)] = 0;
        dev_k2[POS(x, y, data.dis_X, data.dis_Y)] = 0;
        dev_k3[POS(x, y, data.dis_X, data.dis_Y)] = 0;
        dev_k4[POS(x, y, data.dis_X, data.dis_Y)] = 0;
    }
}

/**
 * Kernel for modifying the input data. In Runge-Kutta it is the addition of the difference, which is computed from k1 - k5.
 */
template<typename Traits, typename Data>
__global__ void global_add(typename Traits::real* dev_input, typename Traits::real* dev_u_add, Data data)
{
    using idx = typename Traits::idx;

    idx x = threadIdx.x + (blockIdx.x + data.gridIdx.x*data.grids.x)*blockDim.x;
	idx y = threadIdx.y + (blockIdx.y + data.gridIdx.y*data.grids.y)*blockDim.y;

    dev_input[POS(x,y,data.dis_X,data.dis_Y)] += dev_u_add[POS(x,y,data.dis_X,data.dis_Y)];
}

#endif
#ifndef __TRAITS_H__
#define __TRAITS_H__

/**
 * Macro for computing the location of a 2d point in 1d array.
 */
#define POS(x,y,X,Y) (y+(x)*(Y))

/**
 * Sets up types.
 */
template< typename floatingPoint>
struct Traits
{
    using real = floatingPoint;
    using idx = long int;
    using map_t = short int;
};

using TraitsSP = Traits<float>;
using TraitsDP = Traits<double>;

/**
 * Struct so that setting up the parameters of the task is not confusing.
 */
template< typename Traits>
struct ConstructParam
{
    using real = typename Traits::real;
    using idx = typename Traits::idx;

    real X, Y, time_step, final_time, precision;
    idx dis_X, dis_Y;
    dim3 threadsPerBlock, grids, maxGridSize, blocksCount;

};

#endif
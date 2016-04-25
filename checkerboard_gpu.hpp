#ifndef CHECKERBOARD_GPU
#define CHECKERBOARD_GPU

// Release 0.2
// Adam Levy

#include "checkerboard.hpp"

#include <thrust/device_vector.h>
using thrust::device_vector;

struct BitBoard_gpu
{
    uint32_t red_pos  = RED_INIT_POS_BM;
    uint32_t blk_pos  = BLK_INIT_POS_BM;
    uint32_t king_pos = KING_INIT_POS_BM;
    bool turn = FIRST_TURN;
    __device__ bool operator==( const BitBoard_gpu & ) const;
    __device__ bool operator!=( const BitBoard_gpu & ) const;
    __device__ __host__ BitBoard_gpu & operator=( const BitBoard & );
};

struct BitBoardArray
{
    BitBoard_gpu *  bb_ary = 0;
    size_t      size = 0;
};

__device__ BitBoardArray gen_children_gpu(const BitBoard_gpu & bb);
__device__ BitBoardArray follow_jumps_gpu(const BitBoard_gpu & bb, uint32_t follow_mask = 0xffFFffFF);
__device__ size_t bit_count_gpu(uint32_t i);

#endif // CHECKERBOARD_GPU

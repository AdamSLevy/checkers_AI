#ifndef MCMC_H
#define MCMC_H
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <sstream>
#include <curand_kernel.h>
#include <chrono>
#include <ctime>
#include "checkerboard_gpu.hpp"

#define MAX_MOVES 200

typedef unsigned long long ullong;

using std::endl;
using std::cout;

// Error handling/*{{{*/
// Adapted from the CUDNN classification code 
// sample: https://developer.nvidia.com/cuDNN

#define FatalError(s) do {                                             \
    std::stringstream _where, _message;                                \
    _where << __FILE__ << ':' << __LINE__;                             \
    _message << std::string(s) + "\n" << __FILE__ << ':' << __LINE__;  \
    std::cerr << _message.str() << "\nAborting...\n";                  \
    cudaDeviceReset();                                                 \
    exit(1);                                                           \
} while(0)

#define checkCUDNN(status) do {                                        \
    std::stringstream _error;                                          \
    if (status != CUDNN_STATUS_SUCCESS) {                              \
      _error << "CUDNN failure: " << cudnnGetErrorString(status);      \
      FatalError(_error.str());                                        \
    }                                                                  \
} while(0)

#define checkCudaErrors(status) do {                                   \
    std::stringstream _error;                                          \
    if (status != 0) {                                                 \
      _error << "Cuda failure: " << status;                            \
      FatalError(_error.str());                                        \
    }                                                                  \
} while(0)/*}}}*/

using std::chrono::system_clock;

__global__ void setup_kernel(curandState *state, ullong r_offset);
__global__ void random_descent( curandState * state, 
                                BitBoard_gpu * d_bb, 
                                ullong * d_wins, bool player);
#endif // MCMC_H

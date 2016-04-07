#include <cudnn.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <stdio.h>

#include <sys/stat.h>

using std::endl;
using std::cout;

//////////////////////////////////////////////////////////////////////////////
// Error handling
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
} while(0)

bool fileExists(const char* file) {
    struct stat buf;
    return (stat(file, &buf) == 0);
}

#define RAW_BOARD_BYTES    (4*3)
#define BOARD_TENSOR_FLOATS (8*8*3)

#include "checkerboard.hpp"

void printBoardTensor(float * boardTensor)
{
    for (int c = 0; c < 3; c++){
        string board_out;
        for (int h = 0; h < 8; h++){
            string row;
            for (int w = 0; w < 8; w++){
                int float_id = c * 8 * 8 + h * 8 + w;
                float ff = boardTensor[float_id];
                string value = " *** ";
                if (ff == 1.0f){
                    value = " 1.0 ";
                }
                row += value;
            }
            row += "\n";
            board_out = row + board_out;
        }
        cout << board_out << endl;
    }
}

__constant__ uint32_t POS_MASK_D[32];

__global__ void raw_game_to_tensor(uint32_t * raw_game, float * game_tensor, size_t num_boards)
{
    size_t board_id = blockIdx.x * (blockDim.x * blockDim.y) + threadIdx.x;     // A board consists of 3 uint32_t bitboards
    if (board_id >= num_boards){
        return;
    }

    size_t bitboard_id = board_id * 3 + threadIdx.y;
    size_t tensor_id_start = bitboard_id * (8 * 8);      // A uint32_t bitboard translates to an 8*8 sparse matrix of floats

    uint32_t board = raw_game[bitboard_id];

    for (size_t bit = 0; bit < 32; bit++){
        size_t fvalue_id;
        size_t fzero_id;
        if ((bit / 4) % 2 == 0){
            fvalue_id = bit * 2;
            fzero_id  = bit * 2 + 1;
            
        } else{
            fvalue_id = bit * 2 + 1;
            fzero_id  = bit * 2;
        }
        float value = 0.0f;
        if (board & POS_MASK_D[bit]){
            value = 1.0f;
        }
        game_tensor[tensor_id_start + fvalue_id] = value;
        game_tensor[tensor_id_start + fzero_id]  = 0.0f;
    }
}

int main()
{
    // READ IN FILE
    std::ifstream infile;

    // Create file name
    static size_t file_num = 1;
    char str[40];
    sprintf(str, "./game_data/games%04lu.bin", file_num++);

    // Check for file.
    if (!fileExists(str)){
        sprintf(str, "./game_data/games%04lu.bin does not exist.", file_num);
        cout << str << endl;
        cout << "Exiting" << endl;
        exit(1);
    }

    // Open and error checking
    infile.open(str, std::ios::in | std::ios::binary);
    if (!infile.is_open()){
        cout << "Failed to open file" << endl;
        cout << "Exiting" << endl;
        exit(1);
    } else{
        cout << str << endl;
    }

    // Get file length
    infile.seekg (0, infile.end);
    int length = infile.tellg();
    infile.seekg (0, infile.beg);

    // Verify valid length
    size_t num_uint = length / sizeof(uint32_t);
    if (length % sizeof(uint32_t) != 0 ||
            num_uint % 3 != 0){
        cout << "Invalid input file size" << endl;
        exit(1);
    }

    // Read in data
    uint32_t raw_game[num_uint];
    infile.read((char *) raw_game, length);

    infile.close();

    // Allocate mem on device
    uint32_t * d_raw_game;
	checkCudaErrors(cudaMalloc(&d_raw_game, num_uint * sizeof(uint32_t)));

    // Copy raw game data to device
    checkCudaErrors(cudaMemcpy(d_raw_game, raw_game, num_uint * sizeof(uint32_t), cudaMemcpyHostToDevice));

    // Allocate mem for game tensor
    int n_boards, c_bitboards, h_rows, w_cols;
    c_bitboards = 3;
    n_boards = num_uint / c_bitboards;
    h_rows = 8;
    w_cols = 8;
    size_t num_game_tensor_floats = num_uint * h_rows * w_cols;
    float * d_game_tensor;
	checkCudaErrors(cudaMalloc(&d_game_tensor, num_game_tensor_floats * sizeof(float)));

    // Copy POS_MASK
	checkCudaErrors(cudaMemcpyToSymbol(POS_MASK_D, POS_MASK, 32 * sizeof(uint32_t), size_t(0), cudaMemcpyHostToDevice));

    // Generate game tensor
    size_t num_blocks = num_uint / 1024 + 1;
    dim3 threadsPerBlock(1024/3, 3);
    raw_game_to_tensor<<<num_blocks,threadsPerBlock>>>(d_raw_game, d_game_tensor, n_boards);
	checkCudaErrors(cudaDeviceSynchronize());

    // Copy game tensor data to host
    size_t nb = 3;
    float boardTensor[BOARD_TENSOR_FLOATS * nb];
    checkCudaErrors(cudaMemcpy(boardTensor, d_game_tensor, nb * BOARD_TENSOR_FLOATS * sizeof(float), cudaMemcpyDeviceToHost));

    for(int i = 0; i < nb; i++){
        printBoardTensor(boardTensor + i * BOARD_TENSOR_FLOATS);
    }

    // Free memory
	checkCudaErrors(cudaFree(d_raw_game));
	checkCudaErrors(cudaFree(d_game_tensor));

    exit(0);

    size_t version = cudnnGetVersion();
    if(version/1000 != 4){
        cout << "Not cuDNN v4" << endl;
        cout << "version: " << version << endl;
    }

    cudnnHandle_t handle;
    checkCUDNN(cudnnCreate(&handle));

    cudnnTensorDescriptor_t tensor;

    checkCUDNN(cudnnCreateTensorDescriptor(&tensor));
    checkCUDNN(cudnnSetTensor4dDescriptor(tensor,
                                            CUDNN_TENSOR_NCHW,
                                            CUDNN_DATA_FLOAT,
                                            n_boards,
                                            c_bitboards,
                                            h_rows,
                                            w_cols));


    
    checkCUDNN(cudnnDestroyTensorDescriptor(tensor));
    checkCUDNN(cudnnDestroy(handle));

    return 0;
}
/*
Notes:

Functions:
    cudnnGetVersion()
    cudnnCreate(cudnnHandle_t *)
    cudnnDestroy(cudnnHandle_t)

    cudnnCreateTensorDescriptor(cudnnTensorDescriptor_t *)
    cudnnSetTensor4dDescriptor(cudnnTensorDescriptor_t,
                               cudnnTensorFormat_t       CUDNN_TENSOR_NCHW,
                               cudnnDataType_t           CUDNN_DATA_FLOAT,
                               int                       n_boards,
                               int                       c_bitboards,
                               int                       h_rows,
                               int                       w_cols)
    

*/

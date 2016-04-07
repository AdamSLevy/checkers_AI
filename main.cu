#include <cudnn.h>/*{{{*/

#include <iostream>
#include <fstream>
#include <sstream>
#include <stdio.h>

#include <sys/stat.h>

using std::endl;
using std::cout;/*}}}*/

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

bool fileExists(const char* file) {
    struct stat buf;
    return (stat(file, &buf) == 0);
}

#define RAW_BOARD_BYTES    (4*3)
#define BOARD_TENSOR_FLOATS (8*8*3)

#include "checkerboard.hpp"

void printBoardTensor(float * boardTensor)/*{{{*/
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
}/*}}}*/

__constant__ uint32_t POS_MASK_D[32];

#pragma pack(push, 1)
struct GameStat/*{{{*/
{
    bool win = false;
    uint16_t num_moves = 0;
};/*}}}*/
#pragma pack(pop)

__global__ void gen_label_tensor(float * label_tensor, bool win)/*{{{*/
{
    size_t num_moves = blockDim.x;
    size_t game_id = threadIdx.x;

    if (game_id == 0){
        label_tensor[0] = 0.0f;
        return;
    }

    float value = ((float)(game_id + 1)) / num_moves;
    if (!win){
        value *= -1.0f;
    }

    label_tensor[game_id] = value;
}/*}}}*/

__global__ void raw_game_to_tensor(uint32_t * raw_game, float * game_tensor, size_t num_boards)/*{{{*/
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
}/*}}}*/

int main()/*{{{*/
{
    // READ IN FILE/*{{{*//*{{{*/
    std::ifstream binfile;
    std::ifstream infofile;

    // Create file name
    static size_t file_num = 2;
    char str1[40];
    char str2[40];
    sprintf(str1, "./game_data/games%04lu.bin", file_num);
    sprintf(str2, "./game_data/games%04lu.info", file_num);

    // Check for file.
    if (!fileExists(str1) || !fileExists(str2)){
        sprintf(str1, "./game_data/games%04lu.bin does not exist.", file_num);
        sprintf(str2, " or ./game_data/games%04lu.info does not exist.", file_num);
        cout << str1 << endl;
        cout << str2 << endl;
        cout << "Exiting" << endl;
        exit(1);
    }

    // Open and error checking
    binfile.open(str1, std::ios::in | std::ios::binary);
    infofile.open(str2, std::ios::in | std::ios::binary);
    if (!binfile.is_open() || !infofile.is_open()){
        cout << "Failed to open file" << endl;
        cout << "Exiting" << endl;
        exit(1);
    } else{
        cout << str1 << endl;
        cout << str2 << endl;
    }

    // Get file binlength
    binfile.seekg (0, binfile.end);
    int binlength = binfile.tellg();
    binfile.seekg (0, binfile.beg);

    infofile.seekg (0, infofile.end);
    int infolength = infofile.tellg();
    infofile.seekg (0, infofile.beg);

    // Verify valid binlength
    size_t num_uint = binlength / sizeof(uint32_t);
    if (binlength % sizeof(uint32_t) != 0 ||
            num_uint % 3 != 0){
        cout << "Invalid bin file size" << endl;
        exit(1);
    }

    if (infolength / sizeof(GameStat) != 100){
        cout << infolength << endl;
        cout << "Invalid info file size" << endl;
        exit(1);
    }
    /*}}}*/

    // Read in data
    uint32_t raw_game[num_uint];
    binfile.read((char *) raw_game, binlength);
    binfile.close();

    GameStat gstat[100];
    infofile.read((char *) gstat, infolength);
    infofile.close();
    /*}}}*/

    // Allocate mem on device/*{{{*/
    uint32_t * d_raw_game;
	checkCudaErrors(cudaMalloc(&d_raw_game, num_uint * sizeof(uint32_t)));
    /*}}}*/

    // Copy raw game data to device/*{{{*/
    checkCudaErrors(cudaMemcpy(d_raw_game, raw_game, num_uint * sizeof(uint32_t), cudaMemcpyHostToDevice));
    /*}}}*/

    // Allocate mem for game tensor and label tensor/*{{{*/
    int n_boards, c_bitboards, h_rows, w_cols;
    c_bitboards = 3;
    n_boards = num_uint / c_bitboards;
    h_rows = 8;
    w_cols = 8;
    size_t num_game_tensor_floats = num_uint * h_rows * w_cols;
    float * d_game_tensor;
    float * d_label_tensor;
	checkCudaErrors(cudaMalloc(&d_game_tensor, num_game_tensor_floats * sizeof(float)));
	checkCudaErrors(cudaMalloc(&d_label_tensor, n_boards * sizeof(float)));
    /*}}}*/

    // Copy pos_mask/*{{{*/
	checkCudaErrors(cudaMemcpyToSymbol(POS_MASK_D, POS_MASK, 32 * sizeof(uint32_t), size_t(0), cudaMemcpyHostToDevice));
    /*}}}*/

    // generate game and label tensor/*{{{*/
    size_t num_blocks = num_uint / 1024 + 1;
    dim3 threadsPerBlock(1024/3, 3);
    raw_game_to_tensor<<<num_blocks,threadsPerBlock>>>(d_raw_game, d_game_tensor, n_boards);

    size_t num_moves = 0;
    for (size_t i = 0; i < 100; i++){
        float * start_label = d_label_tensor + num_moves;
        num_moves += gstat[i].num_moves;
        gen_label_tensor<<<1,gstat[i].num_moves>>>(start_label, gstat[i].win);
    }

	checkCudaErrors(cudaDeviceSynchronize());
    /*}}}*/

    // Copy game tensor data to host/*{{{*/
    size_t nb = gstat[0].num_moves;
    float boardTensor[BOARD_TENSOR_FLOATS * nb];
    float labels[nb];
    checkCudaErrors(cudaMemcpy(boardTensor, d_game_tensor, nb * BOARD_TENSOR_FLOATS * sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(labels, d_label_tensor, nb * sizeof(float), cudaMemcpyDeviceToHost));
    /*}}}*/

    // Print out tensor board/*{{{*/
    for(int i = 0; i < nb; i++){
        printBoardTensor(boardTensor + i * BOARD_TENSOR_FLOATS);
    }

    size_t i = 0;
    for(float f : labels)
    {
        cout << i++ << ": " << f << endl;
    }
    /*}}}*/

    // Free memory/*{{{*/
	checkCudaErrors(cudaFree(d_raw_game));
	checkCudaErrors(cudaFree(d_game_tensor));
    /*}}}*/

    exit(0);

    // CUDNN SCRATCH/*{{{*/
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
    /*}}}*/

    return 0;
}/*}}}*/

// Notes:/*{{{*/
/*
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
/*}}}*/

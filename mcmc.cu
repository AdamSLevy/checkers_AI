#include "mcmc.h"

__constant__ uint32_t POS_MASK_D[32];

__global__ void setup_kernel(curandState *state, ullong r_offset)
{
    ullong idx = threadIdx.x + blockDim.x * (blockIdx.x * gridDim.y + blockIdx.y);
    ullong sequence = threadIdx.x;
    ullong seed = (idx + 1) * r_offset;
    curand_init(seed, sequence, r_offset, &state[idx]);
}

__global__ void random_descent( curandState * state, 
        BitBoard_gpu * d_bb, ullong * d_wins, bool player)
{
    __shared__ ullong wins;
    if (threadIdx.x == 0){
        wins = 0;
    }
    __syncthreads();

    int idx = threadIdx.x + blockDim.x * (blockIdx.x * gridDim.y + blockIdx.y);
    curandState localState = state[idx];

    BitBoard_gpu bb = *(d_bb + blockIdx.x);

    auto children = gen_children_gpu(bb);
    size_t n_moves = 0;
    float frand;
    while(children.size && n_moves < MAX_MOVES){
        n_moves++;
        size_t b = children.size;
        frand = curand_uniform(&localState);
        int irand = frand * b;

        bb = children.bb_ary[irand];
        delete [] children.bb_ary;
        children = gen_children_gpu(bb);
    }

    bool winner = !bb.turn;
    if (children.size > 0){
        delete [] children.bb_ary;
        winner = !player;
        if (true){
            size_t red_count = bit_count_gpu(bb.red_pos);
            size_t red_king_count = bit_count_gpu(bb.red_pos & bb.king_pos);
            size_t red_score = red_count + red_king_count;

            size_t blk_count = bit_count_gpu(bb.blk_pos);
            size_t blk_king_count = bit_count_gpu(bb.blk_pos & bb.king_pos);
            size_t blk_score = blk_count + blk_king_count;

            if (player == RED){
                if (red_score > blk_score){
                    winner = player;
                }
            } else{
                if (blk_score > red_score){
                    winner = player;
                }
            }
        }
    }
            
    if (winner == player){
        atomicAdd(&wins, 1);
    }

    state[idx] = localState;
    __syncthreads();
    if (threadIdx.x == 0){
        atomicAdd(d_wins + blockIdx.x, wins);
    }
}

/*
int main(int argc, char *argv[])
{
    int num_repeat = 15;
    int num_plays = 1024;

    if(argc == 2){
        num_repeat = atoi(argv[1]);
    }
    printf("%d repeat\n", num_repeat);

    BitBoard board;
    BitBoard_gpu * d_board;
    auto children = gen_children(board);
    int size = children.size();
    curandState * d_state;
    checkCudaErrors(cudaMalloc((void **)&d_state, size * num_repeat * num_plays * sizeof(curandState)));
    unsigned long long time = system_clock::to_time_t(system_clock::now());
    checkCudaErrors(cudaDeviceSynchronize());
    dim3 blocks(size,num_repeat);
    setup_kernel<<<blocks,num_plays>>>(d_state, time);

    checkCudaErrors(cudaMalloc(&d_board, size * sizeof(curandState)));
    checkCudaErrors(cudaMemcpy(d_board, &children[0], size * sizeof(BitBoard), cudaMemcpyHostToDevice));

    unsigned long long * d_wins;
    checkCudaErrors(cudaMalloc(&d_wins, size * sizeof(unsigned long)));
    checkCudaErrors(cudaMemset(d_wins, 0, size * sizeof(unsigned long)));

    checkCudaErrors(cudaDeviceSynchronize());
    cout << "Calling random_descent" << endl;
    for (int i = 0; i < 10; i++){
        cout << "i: " << i << endl;
        random_descent<<<blocks,num_plays>>>(d_state, d_board, d_wins);
        checkCudaErrors(cudaDeviceSynchronize());
    }

    unsigned long long wins[size];
    checkCudaErrors(cudaMemcpy(wins, d_wins, size * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaDeviceSynchronize());

    for (auto w : wins){
        cout << w << endl;
        cout << (double)w/(num_plays*num_repeat*10) << endl;
    }

    cudaFree(d_state);
    cudaFree(d_wins);
    cudaFree(d_board);
}
*/

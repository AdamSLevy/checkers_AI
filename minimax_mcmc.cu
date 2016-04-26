#include "minimax_mcmc.hpp"

Minimax_mcmc::Minimax_mcmc(BitBoard bb, size_t depth)
    : root_node(bb), max_depth(depth)
{
    // Set up randstate
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaMalloc((void **)&d_state, 
                        MAX_NUM_MOVES * MAX_NUM_REPEAT * NUM_ITERS * sizeof(curandState)));
    checkCudaErrors(cudaDeviceSynchronize());
    ullong time = system_clock::to_time_t(system_clock::now());
    dim3 blocks(MAX_NUM_MOVES, MAX_NUM_REPEAT);
    setup_kernel<<<blocks,NUM_ITERS>>>(d_state, time);
    checkCudaErrors(cudaDeviceSynchronize());

    // Set up space for boards to be evaluated
    checkCudaErrors(cudaMalloc(&d_board, MAX_NUM_MOVES * sizeof(BitBoard_gpu)));
    checkCudaErrors(cudaDeviceSynchronize());

    // Set up space and zero win count
    checkCudaErrors(cudaMalloc(&d_wins, MAX_NUM_MOVES * sizeof(ullong)));
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaMemset(d_wins, 0, MAX_NUM_MOVES * sizeof(ullong)));
    checkCudaErrors(cudaDeviceSynchronize());
    cout << "sizeof(curandState) " << sizeof(curandState) << endl;
}

Minimax_mcmc::~Minimax_mcmc()
{
    cudaFree(d_state);
    cudaFree(d_wins);
    cudaFree(d_board);
}

BitBoard Minimax_mcmc::evaluate(size_t depth)
{
    num_pruned = 0;
    num_visited = 0;
    if (depth == 0){
        depth = max_depth;
    }

    // Get ordered children
    //cout << "first playout" << endl;
    vector<BitBoard> children = mcmc_playout(root_node);
    int num_children = children.size();

    if(num_children == 0){
        return root_node;
    } else if(num_children == 1){
        return children.front();
    }

    // Reset rand states
    //ullong time = system_clock::to_time_t(system_clock::now());
    //dim3 blocks(num_children, MAX_NUM_REPEAT);
    //setup_kernel<<<blocks,NUM_ITERS>>>(d_state, time);

    ullong alpha = std::numeric_limits<ullong>::min();
    ullong  beta = std::numeric_limits<ullong>::max();
    ullong value = std::numeric_limits<ullong>::min();
    //ullong best_value = value;
    vector<ullong> all_values;

    BitBoard best_move = children.back();

    BitBoard zero;
    zero.red_pos = 0;
    zero.blk_pos = 0;
    zero.king_pos = 0;

    for (BitBoard cc : children){
        num_visited++;
        int old_value = value;
        value = std::max(value,
                alphabeta(cc, depth - 1, alpha, beta, root_node, zero));
        all_values.push_back(value);
        alpha = std::max(alpha, value);
        //print_bb(cc);
        if (value != old_value){
            best_move = cc;
            //best_value = value;
        }
        if (beta <= alpha){
            num_pruned++;
            break;
        }
    }

    for (auto v : all_values){
        cout << v << endl;
    }
    cout << endl;

    return best_move;
}


void Minimax_mcmc::set_root_node(BitBoard bb)
{
    root_node = bb;
}

void Minimax_mcmc::set_max_depth(size_t depth)
{
    if (depth > 0){
        max_depth = depth;
    }
}


ullong Minimax_mcmc::alphabeta(const BitBoard & bb, size_t depth, ullong alpha, ullong beta, const BitBoard & parent, const BitBoard & gparent)
{
    // ordered children
    vector<BitBoard> children = mcmc_playout(bb);
    bool maximizing = (bb.turn == root_node.turn);
    if (depth == 0 || children.empty()){
        ullong win_count;
        if (children.empty()){
            if (maximizing){
                win_count = 0;
            } else{
                win_count = std::numeric_limits<ullong>::max();
            }
        } else{
            int num_children = children.size();
            if (maximizing){
                win_count = h_wins[num_children-1];
            } else{
                win_count = h_wins[0];
            }
        }

        return win_count;
    }

    if (maximizing){
        ullong value = std::numeric_limits<ullong>::min();
        for (int i = children.size() - 1; i >= 0; i--){
            BitBoard cc = children[i];
            num_visited++;
            if ((cc.king_pos ^ bb.king_pos) && (parent.king_pos ^ gparent.king_pos)){
                if (bb.turn == BLK){
                    if ((cc.blk_pos & cc.king_pos) == (gparent.blk_pos & gparent.king_pos)){
                        continue;
                    }
                } else{
                    if ((cc.red_pos & cc.king_pos) == (gparent.red_pos & gparent.king_pos)){
                        continue;
                    }
                }
            }
            value = std::max(value,
                    alphabeta(cc, depth - 1, alpha, beta, bb, parent));
            alpha = std::max(alpha, value);
            //print_bb(cc);
            if (beta <= alpha){
                num_pruned++;
                break;
            }
        }
        return value;
    } else{
        ullong value = std::numeric_limits<ullong>::max();
        for (BitBoard cc : children){
            num_visited++;
            if ((cc.king_pos ^ bb.king_pos) && (parent.king_pos ^ gparent.king_pos)){
                if (bb.turn == BLK){
                    if ((cc.blk_pos & cc.king_pos) == (gparent.blk_pos & gparent.king_pos)){
                        continue;
                    }
                } else{
                    if ((cc.red_pos & cc.king_pos) == (gparent.red_pos & gparent.king_pos)){
                        continue;
                    }
                }
            }
            value = std::min(value,
                    alphabeta(cc, depth - 1, alpha, beta, bb, parent));
            beta = std::min(beta, value);
            //print_bb(cc);
            if (beta <= alpha){
                num_pruned++;
                break;
            }
        }
        return value;
    }
}

vector<BitBoard> Minimax_mcmc::mcmc_playout(const BitBoard & bb, size_t num_ki)
{
    auto children = gen_children(bb);
    int num_children = children.size();

    if (num_children <= 1){
        return children;
    }

    // Copy child boards to device
    checkCudaErrors(cudaMemcpy(d_board,
                        &children[0],
                        num_children * sizeof(BitBoard),
                        cudaMemcpyHostToDevice));

    // Zero out win counts
    checkCudaErrors(cudaMemset(d_wins, 0, num_children * sizeof(ullong)));

    dim3 blocks(num_children,num_ki);
    random_descent<<<blocks,NUM_ITERS>>>(d_state, d_board, d_wins, root_node.turn);
    checkCudaErrors(cudaDeviceSynchronize());

    // Sort boards by win count, lowest to highest
    checkCudaErrors(cudaMemcpy(h_wins,
                        d_wins,
                        num_children * sizeof(ullong),
                        cudaMemcpyDeviceToHost));

    thrust::sort_by_key(h_wins, h_wins + num_children, &children[0]);

    for(int i = 0; i < num_children; i++){
        cout << h_wins[i] << endl;
    }
    cout << endl;

    return children;
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

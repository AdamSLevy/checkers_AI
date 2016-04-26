#ifndef MINIMAX_MCMC_H
#define MINIMAX_MCMC_H

#include "checkerboard.hpp"
#include "mcmc.h"
#include <thrust/sort.h>
#include <armadillo>

using arma::mat;

#define MAX_DEPTH 5
#define MAX_NUM_MOVES BB_PRE_ALLOC
#define MAX_NUM_REPEAT 1
#define NUM_ITERS 1024

class Minimax_mcmc
{
    public:
        Minimax_mcmc(BitBoard bb, size_t depth = MAX_DEPTH);
        ~Minimax_mcmc();

        BitBoard evaluate(size_t depth = 0);  // depth = 0, defaults to max_depth

        void set_root_node(BitBoard bb);
        void set_max_depth(size_t depth);
        size_t get_num_pruned(){return num_pruned;}
        size_t get_num_visited(){return num_visited;}

    private:
        BitBoard root_node;
        size_t max_depth;
        size_t num_pruned;
        size_t num_visited;

        BitBoard_gpu * d_board;
        curandState * d_state;
        ullong * d_wins;
        ullong h_wins[MAX_NUM_MOVES];

        // num_ki == number of 1024 playouts per child board
        vector<BitBoard> mcmc_playout(const BitBoard & bb, size_t num_ki = 1); 

        ullong alphabeta(const BitBoard & bb, size_t depth, ullong alpha, ullong beta, const BitBoard & parent, const BitBoard & gparent);
};

#endif // MINIMAX_MCMC_H

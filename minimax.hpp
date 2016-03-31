#ifndef MINIMAX_HPP
#define MINIMAX_HPP

#include "checkerboard.hpp"
#include "ffnn.hpp"

#define MAX_DEPTH 5

class Minimax
{
    public:
        Minimax();
        Minimax(BitBoard bb, FFNN e_nn, size_t depth = MAX_DEPTH)
            : root_node(bb), eval_nn(e_nn), max_depth(depth){};
        Minimax(CheckerBoard cb, FFNN e_nn, size_t depth = MAX_DEPTH)
            : root_node(cb.m_bb), eval_nn(e_nn), max_depth(depth){};

        BitBoard evaluate(size_t depth = 0);                // depth = 0, defaults to max_depth

        void set_root_node(BitBoard bb);
        void set_max_depth(size_t depth);
        size_t get_num_pruned(){return num_pruned;}
        size_t get_num_visited(){return num_visited;}

    private:
        BitBoard root_node;
        FFNN eval_nn;
        size_t max_depth;
        size_t num_pruned;
        size_t num_visited;

        double alphabeta(BitBoard bb, size_t depth, double alpha, double beta);
};

#endif

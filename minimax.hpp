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
            : root_node(CheckerBoard(bb)), eval_nn(e_nn), max_depth(depth){};
        Minimax(CheckerBoard cb, FFNN e_nn, size_t depth = MAX_DEPTH)
            : root_node(cb), eval_nn(e_nn), max_depth(depth){};

        BitBoard evaluate(size_t depth = 0);                // depth = 0, defaults to max_depth

        void set_board(BitBoard bb);
        void set_max_depth(size_t depth);

    private:
        CheckerBoard root_node;
        FFNN eval_nn;
        size_t max_depth;

        double alphabeta(BitBoard bb, size_t depth, double alpha, double beta);
};

#endif

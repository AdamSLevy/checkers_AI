#ifndef MINIMAX
#define MINIMAX

#include "checkerboard.hpp"
#include "static_eval.hpp"
#include <armadillo>

using arma::mat;

#define MAX_DEPTH 5

class Minimax_static
{
    public:
        Minimax_static();
        Minimax_static(BitBoard bb, size_t depth = MAX_DEPTH)
            : root_node(bb), max_depth(depth){};
        Minimax_static(CheckerBoard cb, size_t depth = MAX_DEPTH)
            : root_node(cb.m_bb), max_depth(depth){};

        BitBoard evaluate(size_t depth = 0);                // depth = 0, defaults to max_depth

        void set_root_node(BitBoard bb);
        void set_max_depth(size_t depth);
        size_t get_num_pruned(){return num_pruned;}
        size_t get_num_visited(){return num_visited;}

    private:
        BitBoard root_node;
        vector<uint32_t> cycle_detect[2];
        size_t max_depth;
        size_t num_pruned;
        size_t num_visited;

        int alphabeta(const BitBoard & bb, size_t depth, int alpha, int beta, const BitBoard & parent, const BitBoard & gparent);
};

#endif

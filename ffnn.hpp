#include "checkerboard.h"
#include <armadillo>

using namespace arma;

// Feed Forward Neural Net
class FFNN
{
    public:
        FFNN(vector<size_t> net_layout);

        vector<mat> get_weights(size_t layer_id);

    private:
        vector<mat> weight_mats;
};


// number of layers
// nodes per layer
//
// vector of num_nodes
//
// 32 -> 100 -> 10 -> 1
//
// 32 x 100
//
// 100 x 10
//
// 10 x 1

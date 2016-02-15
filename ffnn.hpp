#include <vector>
using std::vector;

#include <armadillo>
using arma::mat;
using arma::fill::randn;
//using arma::fill::ones;

#include "ThrowAssert.hpp"


// Feed Forward Neural Net
class FFNN
{
    public:
        FFNN(vector<size_t> net_layout);            // net_layout specifies the number of nodes 
                                                    // in each layer. 
                                                    // e.g. (32,100,20,1) creates a 4 layer net
                                                    // with 32 inputs and 1 output
                                                    // 32 -> 100 -> 20 -> 1

        vector<mat> weights();                  // get all weight mats
        mat         weights(size_t layer_id);   // get weight mat for layer_id
        size_t      num_layers();
        size_t      num_nodes(size_t layer_id);
        mat         input();
        mat         output();

        void        set_input(mat in_mat);

        void        forward_pass();
        mat         forward_pass(mat input);

    private:
        vector<mat> weight_mats_;
        mat         input_mat_;
        mat         output_mat_;
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

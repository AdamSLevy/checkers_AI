#include "ffnn.hpp"

FFNN::FFNN(vector<size_t> net_layout)
{
    for (size_t i = 0; i < net_layout.size() - 1; i++){
        size_t num_rows = net_layout[i];
        size_t num_cols = net_layout[i+1];

        weight_mats_.push_back(mat(num_rows, num_cols, randn));
    }
    input_mat_ = mat(1,net_layout.front());
    output_mat_ = mat(1,net_layout.back());
}

vector<mat> FFNN::weights()
{
    return weight_mats_;
}

mat FFNN::weights(size_t layer_id)
{
    throw_assert(layer_id < weight_mats_.size(), 
            "OUT OF BOUNDS, layer_id: " << layer_id 
            << " !< num_layers: " << weight_mats_.size());

    return weight_mats_[layer_id];
}

size_t FFNN::num_layers()
{
    return weight_mats_.size();
}

size_t FFNN::num_nodes(size_t layer_id)
{
    throw_assert(layer_id < weight_mats_.size(), 
            "OUT OF BOUNDS, layer_id: " << layer_id 
            << " !< num_layers: " << weight_mats_.size());

    return weight_mats_[layer_id].size();
}

mat FFNN::input()
{
    return input_mat_;
}

mat FFNN::output()
{
    return output_mat_;
}

void FFNN::set_input(mat in_mat)
{
    throw_assert(in_mat.n_cols == weight_mats_[0].n_rows, 
            "INVALID COL SIZE: in_mat.n_cols: " << in_mat.n_cols 
            << " != weight_mats_[0].n_rows: " << weight_mats_[0].n_rows);

    input_mat_ = in_mat;
}

void FFNN::forward_pass()
{
    mat aa = input_mat_;
    for (mat ww : weight_mats_){
        aa = tanh( aa * ww ); //aa * ww;
    }
    output_mat_ = aa;
}

mat FFNN::forward_pass(mat input)
{
    mat aa = input;
    for (mat ww : weight_mats_){
        aa = tanh( aa * ww ); //aa * ww;
    }
    return aa;
}

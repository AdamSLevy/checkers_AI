#include "ffnn.hpp"

FFNN::FFNN(vector<size_t> net_layout)
{
    for (size_t i = 0; i < net_layout.size() - 1; i++){
        size_t num_rows = net_layout[i];
        size_t num_cols = net_layout[i+1];

        weight_mats.push_back(mat(num_rows, num_cols, fill::randn));
    }
}

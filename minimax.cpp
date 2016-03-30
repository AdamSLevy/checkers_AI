#include "minimax.hpp"

BitBoard Minimax::evaluate(size_t depth)
{
    if (depth == 0){
        depth = max_depth;
    }

    vector<BitBoard> children = gen_children(root_node);

    // put the children in order according to ffnn
    mat values = eval_nn.forward_pass(gen_input_mat(children));
    arma::uvec order = sort_index(values.t(), "descend");

    vector<BitBoard> ordered_children(children.size());
    for (size_t i = 0; i < children.size(); i++){
        ordered_children[i] = children[order[i]];
    }

    double alpha = std::numeric_limits<double>::min();
    double  beta = std::numeric_limits<double>::max();
    double value = std::numeric_limits<double>::min();

    BitBoard best_move = ordered_children.front();

    for (BitBoard bb : ordered_children){
        double old_value = value;
        value = std::max(value,
                alphabeta(bb, depth - 1, alpha, beta));
        print_bb(bb);
        if (value != old_value){
            best_move = bb;
        }

        alpha = std::max(alpha, value);
        if (beta <= alpha){
            break;
        }
    }
    
    return best_move;
}


void Minimax::set_board(BitBoard bb)
{
    root_node = bb;
}


void Minimax::set_max_depth(size_t depth)
{
    if (depth > 0){
        max_depth = depth;
    }
}


double Minimax::alphabeta(BitBoard bb, size_t depth, double alpha, double beta)
{
    vector<BitBoard> children = gen_children(bb);

    // order children
    if (depth == 0 || children.empty()){
        return eval_nn.forward_pass(gen_input_mat(bb))(0);
    }

    // put the children in order according to ffnn
    mat values = eval_nn.forward_pass(gen_input_mat(children));
    arma::uvec order;

    bool maximizing = (bb.turn == root_node.turn);

    if (maximizing){
        order = sort_index(values.t(), "descend");
    } else{
        order = sort_index(values.t()); // ascend
    }

    vector<BitBoard> ordered_children(children.size());
    for (size_t i = 0; i < children.size(); i++){
        ordered_children[i] = children[order[i]];
    }

    if (maximizing){
        double value = std::numeric_limits<double>::min();
        for (BitBoard cc : ordered_children){
            value = std::max(value,
                    alphabeta(cc, depth - 1, alpha, beta));
            alpha = std::max(alpha, value);
            print_bb(cc);
            if (beta <= alpha){
                break;
            }
        }
        return value;
    } else{
        double value = std::numeric_limits<double>::max();
        for (BitBoard cc : ordered_children){
            value = std::min(value,
                    alphabeta(cc, depth - 1, alpha, beta));
            beta = std::min(beta, value);
            print_bb(cc);
            if (beta <= alpha){
                break;
            }
        }
        return value;
    }
}


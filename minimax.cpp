#include "minimax.hpp"

BitBoard Minimax::evaluate(size_t depth)
{
    num_pruned = 0;
    num_visited = 0;
    if (depth == 0){
        depth = max_depth;
    }

    vector<BitBoard> children = gen_children(root_node);
    if(children.size() == 0){
        return root_node;
    } else if(children.size() == 1){
        return children.front();
    }

    // put the children in order according to ffnn
    mat values = eval_nn.forward_pass(gen_input_mat(children, root_node.turn));
    cout << values << endl;
    arma::uvec order = sort_index(values.t(), "descend");

    vector<BitBoard> ordered_children(children.size());
    for (size_t i = 0; i < children.size(); i++){
        ordered_children[i] = children[order[i]];
    }


    double alpha = std::numeric_limits<double>::min();
    double  beta = std::numeric_limits<double>::max();
    double value = std::numeric_limits<double>::min();

    BitBoard best_move = ordered_children.front();

    double best_value;
    vector<double> all_values;
    if (root_node.turn == BLK){
        value = std::numeric_limits<double>::min();
        best_value = value;
        for (BitBoard cc : ordered_children){
            num_visited++;
            double old_value = value;
            value = std::max(value,
                    alphabeta(cc, depth - 1, alpha, beta));
            all_values.push_back(value);
            alpha = std::max(alpha, value);
            //print_bb(cc);
            if (value != old_value){
                best_move = cc;
                best_value = value;
            }
            if (beta <= alpha){
                num_pruned++;
                break;
            }
        }
    } else{
        value = std::numeric_limits<double>::max();
        best_value = value;
        for (BitBoard cc : ordered_children){
            num_visited++;
            double old_value = value;
            value = std::min(value,
                    alphabeta(cc, depth - 1, alpha, beta));
            all_values.push_back(value);
            beta = std::min(beta, value);
            //print_bb(cc);
            if (value != old_value){
                best_move = cc;
                best_value = value;
            }
            if (beta <= alpha){
                num_pruned++;
                break;
            }
        }
    }

    cout << "alpha beta vals " << endl;
    for(double d : all_values){
        cout << d << endl;
    }

    string turnString;
    if(root_node.turn == BLK){
        turnString = "BLK";
    } else{
        turnString = "RED";
    }
    
    cout << "turn " << turnString << endl;
    cout << "best value " << best_value << endl;
    return best_move;
}


void Minimax::set_root_node(BitBoard bb)
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
        return eval_nn.forward_pass(gen_input_mat(bb, bb.turn))(0);
    }

    // put the children in order according to ffnn
    mat values = eval_nn.forward_pass(gen_input_mat(children, bb.turn));
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
            num_visited++;
            value = std::max(value,
                    alphabeta(cc, depth - 1, alpha, beta));
            alpha = std::max(alpha, value);
            //print_bb(cc);
            if (beta <= alpha){
                num_pruned++;
                break;
            }
        }
        return value;
    } else{
        double value = std::numeric_limits<double>::max();
        for (BitBoard cc : ordered_children){
            num_visited++;
            value = std::min(value,
                    alphabeta(cc, depth - 1, alpha, beta));
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


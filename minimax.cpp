#include "minimax.hpp"

BitBoard Minimax::evaluate(size_t depth)
{
    if (depth == 0){
        depth = max_depth;
    }

    root_node.gen_children();

    // put the children in order according to ffnn

    double alpha = std::numeric_limits<double>::min();
    double  beta = std::numeric_limits<double>::max();
    double value = std::numeric_limits<double>::min();

    BitBoard best_move = root_node.m_children.front();

    for(BitBoard bb : root_node.m_children){
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
    root_node = CheckerBoard(bb);
}


void Minimax::set_max_depth(size_t depth)
{
    if (depth > 0){
        max_depth = depth;
    }
}


double Minimax::alphabeta(BitBoard bb, size_t depth, double alpha, double beta)
{
    CheckerBoard cb(bb);
    cb.gen_children();
    // order children
    
    if (depth == 0 || cb.m_children.empty()){
        return eval_nn.forward_pass(gen_input_mat(bb))(0);
    }

    if (bb.turn == root_node.m_bb.turn){
        double value = std::numeric_limits<double>::min();
        for (BitBoard cc : cb.m_children){
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
        for (BitBoard cc : cb.m_children){
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



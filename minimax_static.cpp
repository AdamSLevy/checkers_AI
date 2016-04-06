#include "minimax_static.hpp"

BitBoard Minimax_static::evaluate(size_t depth)
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
    mat values(children.size(),1);

    for(size_t i = 0; i < children.size(); i++){
        values(i,0) = static_eval(children[i], root_node.turn);
    }

    arma::uvec order = sort_index(values.t(), "descend");

    vector<BitBoard> ordered_children(children.size());
    for (size_t i = 0; i < children.size(); i++){
        ordered_children[i] = children[order[i]];
    }


    int alpha = std::numeric_limits<int>::min();
    int  beta = std::numeric_limits<int>::max();
    int value = std::numeric_limits<int>::min();

    BitBoard best_move = ordered_children.front();

    BitBoard zero;
    zero.red_pos = 0;
    zero.blk_pos = 0;
    zero.king_pos = 0;

    int best_value;
    vector<int> all_values;
    //if (root_node.turn == BLK){
    value = std::numeric_limits<int>::min();
    best_value = value;
    for (BitBoard cc : ordered_children){
        num_visited++;
        int old_value = value;
        value = std::max(value,
                alphabeta(cc, depth - 1, alpha, beta, root_node, zero));
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
        /*
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

    cout << "initial values" << endl;
    cout << values << endl;
    cout << "alpha beta vals " << endl;
    for(int d : all_values){
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
    */
    return best_move;
}


void Minimax_static::set_root_node(BitBoard bb)
{
    root_node = bb;
}


void Minimax_static::set_max_depth(size_t depth)
{
    if (depth > 0){
        max_depth = depth;
    }
}


int Minimax_static::alphabeta(const BitBoard & bb, size_t depth, int alpha, int beta, const BitBoard & parent, const BitBoard & gparent)
{
    vector<BitBoard> children = gen_children(bb);

    // order children
    if (depth == 0 || children.empty()){
        return static_eval(bb, root_node.turn);
    }

    // put the children in order according to ffnn
    arma::Mat<int> values(children.size(),1);
    for(size_t i = 0; i < children.size(); i++){
        values(i,0) = static_eval(children[i], root_node.turn);
    }

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
        int value = std::numeric_limits<int>::min();
        for (BitBoard cc : ordered_children){
            num_visited++;
            if ((cc.king_pos ^ bb.king_pos) && (parent.king_pos ^ gparent.king_pos)){
                if (bb.turn == BLK){
                    if ((cc.blk_pos & cc.king_pos) == (gparent.blk_pos & gparent.king_pos)){
                        continue;
                    }
                } else{
                    if ((cc.red_pos & cc.king_pos) == (gparent.red_pos & gparent.king_pos)){
                        continue;
                    }
                }
            }
            value = std::max(value,
                    alphabeta(cc, depth - 1, alpha, beta, bb, parent));
            alpha = std::max(alpha, value);
            //print_bb(cc);
            if (beta <= alpha){
                num_pruned++;
                break;
            }
        }
        return value;
    } else{
        int value = std::numeric_limits<int>::max();
        for (BitBoard cc : ordered_children){
            num_visited++;
            if ((cc.king_pos ^ bb.king_pos) && (parent.king_pos ^ gparent.king_pos)){
                if (bb.turn == BLK){
                    if ((cc.blk_pos & cc.king_pos) == (gparent.blk_pos & gparent.king_pos)){
                        continue;
                    }
                } else{
                    if ((cc.red_pos & cc.king_pos) == (gparent.red_pos & gparent.king_pos)){
                        continue;
                    }
                }
            }
            value = std::min(value,
                    alphabeta(cc, depth - 1, alpha, beta, bb, parent));
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


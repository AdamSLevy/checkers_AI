// Release 0.2
// Adam Levy

#include <iostream>

using std::cout;
using std::endl;

#include "minimax.hpp"

int main(){

    vector<size_t> net_lo;
    net_lo.push_back(32);
    net_lo.push_back(100);
    net_lo.push_back(10);
    net_lo.push_back(1);
    FFNN nn(net_lo);


    BitBoard bb;
    Minimax mm(bb, nn, 10);
    size_t moves = 1;

    BitBoard move = mm.evaluate();

    print_bb(bb);

    print_bb(move);
    cout << "num visited " << mm.get_num_visited() << endl;
    cout << "num pruned " << mm.get_num_pruned() << endl;

    bool has_move = true;
    while(has_move){
        moves++;
        mm.set_root_node(move);
        move = mm.evaluate();
        print_bb(move);
        cout << "num visited " << mm.get_num_visited() << endl;
        cout << "num pruned " << mm.get_num_pruned() << endl;
        if (move.turn == BLK){
            has_move = move.blk_pos;
        } else{
            has_move = move.red_pos;
        }
    }
    cout << "moves " << moves << endl;

    return 0;
}

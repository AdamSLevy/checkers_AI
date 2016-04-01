// Release 0.2
// Adam Levy

#include <iostream>

using std::cout;
using std::endl;

#include "minimax_static.hpp"

int main(){
    BitBoard bb;
    Minimax_static mm(bb, 5);
    size_t moves = 1;

    BitBoard move = mm.evaluate();

    cout << "0: " << endl;
    print_bb(bb);

    cout << moves << ": " << endl;
    print_bb(move);
    cout << "num visited " << mm.get_num_visited() << endl;
    cout << "num pruned " << mm.get_num_pruned() << endl;

    bool has_move = true;
    while(has_move){
        moves++;
        int m = 5 + moves / 5;
        mm.set_max_depth(std::min(m, 12));
        mm.set_root_node(move);
        move = mm.evaluate();
        cout << moves << ": " << endl;
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

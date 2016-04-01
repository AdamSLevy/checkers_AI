// Release 0.2
// Adam Levy

#include <iostream>
#include <random>

using std::cout;
using std::endl;

#include "checkerboard.hpp"

typedef std::default_random_engine def_rand_eng;
typedef std::uniform_int_distribution<int> unif_dist;

int main(){
    def_rand_eng gen;

    BitBoard move;
    size_t moves = 1;

    cout << "0: " << endl;
    print_bb(move);

    bool has_move = true;
    while(has_move){
        auto children = gen_children(move);
        unif_dist dist(0,children.size() - 1);
        int rand_index = dist(gen);
        move = children[rand_index];
        moves++;
        cout << moves << ": " << endl;
        print_bb(move);
        if (move.turn == BLK){
            has_move = move.blk_pos;
        } else{
            has_move = move.red_pos;
        }
    }
    cout << "moves " << moves << endl;

    return 0;
}

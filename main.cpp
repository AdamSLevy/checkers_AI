// Release 0.2
// Adam Levy

#include <iostream>
#include <random>
#include <chrono>

using std::cout;
using std::endl;

#include "checkerboard.hpp"
#include "minimax_static.hpp"

typedef std::default_random_engine def_rand_eng;
typedef std::uniform_int_distribution<int> unif_dist;
typedef std::bernoulli_distribution bern_dist;

int main(){
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    def_rand_eng gen(seed);

    BitBoard move;
    Minimax_static mm(move, 15);
    size_t moves = 0;
    size_t rand_moves = 0;

    cout << "0: " << endl;
    print_bb(move);

    bool has_move = true;
    bern_dist bdist(.5);
    for (size_t i = 0; i < 20; i++){
        while (has_move){
            bool pick_rand = bdist(gen);

            // random move
            if (pick_rand){
                auto children = gen_children(move);
                if (children.size()){
                    unif_dist udist(0,children.size() - 1);
                    int rand_index = udist(gen);
                    move = children[rand_index];
                    moves++;
                    rand_moves++;
                    cout << moves << ": " << "rand" << endl;
                    print_bb(move);
                } else{
                    has_move = false;
                }
            } else{     // minimax move
                mm.set_root_node(move);
                BitBoard newMove = mm.evaluate(10);
                if (newMove != move){
                    moves++;
                    move = newMove;
                    cout << moves << ": " << "minimax" << endl;
                    print_bb(move);
                } else{
                    has_move = false;
                }
            }
        }
    }

    cout << "rand_moves " << rand_moves << endl;
    cout << "moves " << moves << endl;

    double p = (double)rand_moves / moves;

    cout << p << endl;

    return 0;
}

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
    Minimax mm(bb, nn, 2);

    BitBoard move = mm.evaluate();

    print_bb(bb);
    print_bb(move);

    return 0;
}

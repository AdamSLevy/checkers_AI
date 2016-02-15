// Release 0.2
// Adam Levy

#include <iostream>

using std::cout;
using std::endl;

#include "checkerboard.hpp"
#include "ffnn.hpp"

int main(){

    vector<size_t> net_lo;
    net_lo.push_back(32);
    net_lo.push_back(100);
    net_lo.push_back(10);
    net_lo.push_back(1);
    FFNN nn(net_lo);

    BitBoard bb;
    mat in = gen_input_mat(bb);

    nn.set_input(in);
    nn.forward_pass();
    mat out = nn.output();

    cout << "in: \n" << in << endl;
    cout << "out: \n" << out << endl;

    /*
    CheckerBoard board(from_string("____r___bbb__R__bbr_r___br__R___",RED));

    size_t num_boards = 5;
    while ( num_boards > 0 ){
        board.gen_children();
        CheckerBoard child;
        if (board.m_children.size() > 0){
            cout << "parent: " << endl;
            print_bb(board.m_bb);
            cout << "children: " << endl;
            size_t i = 1;
            for (auto b : board.m_children){
                cout << i++ << " : " << endl;
                print_bb(b);
                cout << to_string(b) << endl;
            }
            board = CheckerBoard(board.m_children[0]);
        } else{
            break;
        }
        num_boards--;
    }
    */

    return 0;
}

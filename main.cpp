// Release 0.1
// Adam Levy

#include <iostream>

using std::cout;
using std::endl;

#include "checkerboard.h"

int main(){
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

    return 0;
}

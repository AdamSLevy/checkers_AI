// Release 0.1
// Adam Levy

#include <iostream>

using std::cout;
using std::endl;

#include "checkerboard.h"

int main(){
    CheckerBoard board(from_string("rrrrrrrbrbrr_bB_r____bb_bbbbbbbb",BLK));

    size_t num_boards = 20;
    while ( num_boards > 0 ){
        board.gen_children();
        CheckerBoard child;
        if (board.m_children.size() > 0){
            cout << "parent: " << endl;
            print_bb(board.m_bb);
            cout << "children: " << endl;
            for (auto b : board.m_children){
                cout << to_string(board.m_bb) << endl;
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

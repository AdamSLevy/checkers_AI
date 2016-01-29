#include <iostream>

using std::cout;
using std::endl;

#include "checkerboard.h"

int main(){
    CheckerBoard board;

    size_t num_boards = 20;
    while( num_boards > 0 ){
        board.gen_children();
        CheckerBoard child;
        if(board.m_children.size() > 0){
            child = CheckerBoard(board.m_children[0]);
        } else{
            break;
        }
        child.gen_children();
        if(child.m_children.size() > 0){
            board = CheckerBoard(child.m_children[0]);
        } else{
            break;
        }
        num_boards--;
    }

    return 0;
}

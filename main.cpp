#include <iostream>

using std::cout;
using std::endl;

#include "checkerboard.h"

int main(){
    CheckerBoard board;
    board.gen_children();

    CheckerBoard child = CheckerBoard(board.children[0]);
    child.gen_children();

    return 0;
}

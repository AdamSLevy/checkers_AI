#include "bit_mask_init.h"

#include <vector>

using std::vector;

// DEBUG
#include <iostream>
using std::cout;
using std::endl;
// DEBUG

struct Piece
{
    bool    is_red  = false;
    bool    is_king = false;
    size_t  pos_index;
};

class CheckerBoard
{
    public:
        CheckerBoard();
        void gen_children();

        void test_is_occupied();
    private:
        //CheckerBoard(const CheckerBoard &);
        //operator=(const CheckerBoard &);
        uint32_t red_pos_bm;
        uint32_t blk_pos_bm;
        uint32_t king_pos_bm;

        vector<Piece> red_pieces;
        vector<Piece> blk_pieces;

        vector<CheckerBoard> children;

        bool is_occupied(size_t pos_index);
};

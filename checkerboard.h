#include "bit_mask_init.h"

#include <vector>

using std::vector;

#include <string>
using std::string;

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


struct BitBoard
{
    uint32_t red_pos  = RED_INIT_POS_BM;
    uint32_t blk_pos  = BLK_INIT_POS_BM;
    uint32_t king_pos = KING_INIT_POS_BM;
    bool turn = FIRST_TURN;
    bool operator==( const BitBoard & ) const;
};

    

class CheckerBoard
{
    public:
        CheckerBoard();
        CheckerBoard(BitBoard);
        void gen_children();

        void test_is_occupied();

        vector<BitBoard> children;
    private:
        //CheckerBoard(const CheckerBoard &);
        //operator=(const CheckerBoard &);
        BitBoard m_bb;

        vector<Piece> red_pieces;
        vector<Piece> blk_pieces;


        bool is_occupied(size_t pos_index);
};


void print_board( const uint32_t board );
void print_bb( const BitBoard & bb);

// Release 0.2
// Adam Levy

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

uint32_t movers(BitBoard const & bb, bool kings = false);
uint32_t move_locations(BitBoard const & bb, bool kings = false);
uint32_t jumpers(BitBoard const & bb, bool kings = false);
uint32_t jump_locations(BitBoard const & bb, bool kings = false);

class CheckerBoard
{
    public:
        CheckerBoard();
        CheckerBoard(BitBoard);
        void gen_children();

        BitBoard m_bb;
        vector<BitBoard> m_children;
    private:
        //CheckerBoard(const CheckerBoard &);
        //operator=(const CheckerBoard &);

        bool is_occupied(size_t pos_index);

        vector<BitBoard> follow_jumps(const BitBoard & bb, uint32_t follow_mask = 0xffFFffFF);
};


void print_board( const uint32_t board );
void print_bb( const BitBoard & bb);
string to_string( const BitBoard & bb);
BitBoard from_string( const string & s_board, bool turn );

#include <armadillo>
using arma::mat;
using arma::rowvec;
rowvec gen_input_mat(const BitBoard & bb);
mat gen_input_mat(const vector<BitBoard> & vec_bb);

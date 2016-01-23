#include "checkerboard.h"

CheckerBoard::CheckerBoard()
{
    // set initial position bitmaps
    red_pos_bm = RED_INIT_POS_BM;
    blk_pos_bm = BLK_INIT_POS_BM;
    king_pos_bm = KING_INIT_POS_BM;

    // populate red/blk piece vector
    size_t blk_piece_pos = LAST_POS_INDEX;
    for( size_t red_piece_pos = 0; red_piece_pos < NUM_PIECES; red_piece_pos++ ){
        Piece piece;

        // red piece
        piece.is_red = true;
        piece.pos_index = red_piece_pos;
        red_pieces.push_back(piece);

        // blk piece
        piece.is_red = false;
        piece.pos_index = blk_piece_pos--;
        blk_pieces.push_back(piece);
    }
}

bool CheckerBoard::is_occupied( size_t pos_index )
{
    return POS_MASK[pos_index] & (red_pos_bm | blk_pos_bm);
}

void CheckerBoard::test_is_occupied()
{
    for(size_t i = 0; i < NUM_POS; i++){
        cout << is_occupied(i);
    }

}

void CheckerBoard::gen_children()
{

}



#include "checkerboard.h"

CheckerBoard::CheckerBoard()/*{{{*/
{
    // populate red/blk piece vector
    size_t blk_piece_pos = LAST_POS_INDEX;
    for (size_t red_piece_pos = 0; red_piece_pos < NUM_PIECES; red_piece_pos++){
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
}/*}}}*/

CheckerBoard::CheckerBoard(BitBoard bb) : m_bb(bb){};

bool CheckerBoard::is_occupied( size_t pos_index )/*{{{*/
{
    return POS_MASK[pos_index] & (m_bb.red_pos | m_bb.blk_pos);
}/*}}}*/

void CheckerBoard::test_is_occupied()/*{{{*/
{
    for (size_t i = 0; i < NUM_POS; i++){
        cout << is_occupied(i);
    }

}/*}}}*/

void print_board( const uint32_t board )/*{{{*/
{
    string line;
    for (int i = 31; i >= 0; i--){
        bool bit = board & POS_MASK[i];
        string sym = "0";
        if (bit){
            sym = "1";
        }
        line = " " + sym + line;
        if ((i % 8)){
            line = "  " + line;
        }
        if (!(i % 4)){
            cout << line << endl;
            line.clear();
        }
    }
    cout << endl;
}/*}}}*/

void print_bb( const BitBoard & bb)/*{{{*/
{
    string line;
    for (int i = 31; i >= 0; i--){
        bool has_piece = (bb.red_pos | bb.blk_pos) & POS_MASK[i];
        string sym = "*";
        if (has_piece){
            bool is_king = bb.king_pos & POS_MASK[i];
            if (bb.blk_pos & POS_MASK[i]){
                if(is_king){
                    sym = "B";
                } else{
                    sym = "b";
                }
            } else{
                if(is_king){
                    sym = "R";
                } else{
                    sym = "r";
                }
            }
        }
        line = " " + sym + line;
        if ((i % 8)){
            line = "  " + line;
        }
        if (!(i % 4)){
            cout << line << endl;
            line.clear();
        }
    }
    cout << endl;

}/*}}}*/

void CheckerBoard::gen_children()/*{{{*/
{
    cout << "Parent: " << endl;
    print_bb(m_bb);
    uint32_t occupied = (m_bb.red_pos | m_bb.blk_pos);
    //cout << "occupied" << endl;
    //print_board(occupied);

    uint32_t empty = ~occupied;
    //cout << "empty" << endl;
    //print_board(empty);

    uint32_t play_pos;
    uint32_t oppo_pos;

    if (m_bb.turn == BLK){
        play_pos = m_bb.blk_pos;
        oppo_pos = m_bb.red_pos;
    } else{
        play_pos = m_bb.red_pos;
        oppo_pos = m_bb.blk_pos;
    }

    uint32_t valid_move_pieces = BCKWD(m_bb.turn, empty) & play_pos;
    //cout << "valid move pieces" << endl;
    //print_board(valid_move_pieces);
    uint32_t valid_move_kings  = (valid_move_pieces | FORWD(m_bb.turn, empty)) & play_pos & m_bb.king_pos;

    uint32_t move_locations = FORWD(m_bb.turn, valid_move_pieces);
    //cout << "move_locations" << endl;
    //print_board(move_locations);

    uint32_t empty_move_locations    = move_locations & empty;
    //cout << "empty_move_locations" << endl;
    //print_board(empty_move_locations);

    uint32_t occupied_move_locations = move_locations & (~empty);
    //cout << "occupied_move_locations" << endl;
    //print_board(occupied_move_locations);

    // generate jumps of length 1
    uint32_t first_jumps = FORWD(m_bb.turn, occupied_move_locations) & empty;
    //cout << "first_jumps" << endl;
    //print_board(first_jumps);

    cout << "Children: " << endl;
    // generate child boards
    uint32_t moves_remaining = empty_move_locations;
    for (size_t r = 0; r < 8; r++){
        if (ROW_MASK(r) & moves_remaining){
            // MOVES ON THIS ROW
            for (size_t c = 0; c < 4; c++){
                uint32_t move_loc = COL_MASK(c) & ROW_MASK(r) & moves_remaining;
                if (move_loc){                           // MOVE FOUND: a valid move can be made to r,c
                    // find pieces that can move here
                    size_t move_piece[2];
                    move_piece[0] = (((move_loc >> 4) & play_pos) & (0xffFFffFF * m_bb.turn)) | (((move_loc << 4) & play_pos) & (0xffFFffFF * !m_bb.turn));
                    move_piece[1] = BCKWD(m_bb.turn, move_loc) & (~move_piece[0]);
                    
                    for (int p = 0; p < 2; p++){
                        if (move_piece[p]){
                            BitBoard child_board = m_bb;
                            child_board.turn = !m_bb.turn;
                            uint32_t new_play_pos = play_pos & ~(move_piece[p]);
                            new_play_pos |= move_loc;
                            if (m_bb.turn == BLK){
                                child_board.blk_pos = new_play_pos;
                            } else{
                                child_board.red_pos = new_play_pos;
                            }

                            children.push_back(child_board);
                            cout << "Move " << children.size() << ": " << endl;
                            print_bb(child_board);
                        }
                    }
                }
                // uncheck move location from remaining moves
                moves_remaining ^= move_loc;
                if (!(ROW_MASK(r) & moves_remaining)){      // no remaining moves in row
                    break;
                }
            }
        }
    }

}/*}}}*/




bool BitBoard::operator==(const BitBoard &rhs) const {
    return (red_pos == rhs.red_pos &&
            blk_pos == rhs.blk_pos &&
            king_pos == rhs.king_pos &&
            turn == rhs.turn);
}

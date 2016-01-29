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
        string sym = "**";
        if (has_piece){
            bool is_king = bb.king_pos & POS_MASK[i];
            if (bb.blk_pos & POS_MASK[i]){
                if(is_king){
                    sym = "BB";
                } else{
                    sym = "bb";
                }
            } else{
                if(is_king){
                    sym = "RR";
                } else{
                    sym = "rr";
                }
            }
        }
        line = " " + sym + " " + line;
        if ((i % 8)){
            line = "   " + line;
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

    vector<BitBoard> jump_children = follow_jumps(m_bb);

    if (jump_children.size() > 0){
        cout << "JUMPS AVAILABLE" << endl;
        for (auto jc : jump_children){
            m_children.push_back(jc);
            cout << "jump: " << endl;
            print_bb(jc);
        }
        return;
    }

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

                            m_children.push_back(child_board);
                            cout << "Move " << m_children.size() << ": " << endl;
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


vector<BitBoard> CheckerBoard::follow_jumps(const BitBoard & bb, uint32_t follow_mask)
{
    vector<BitBoard> children;
    //cout << "Parent: " << endl;
    //print_bb(bb);

    uint32_t play_pos;
    uint32_t oppo_pos;

    bool turn = bb.turn;

    if (bb.turn == BLK){
        play_pos = bb.blk_pos;
        oppo_pos = bb.red_pos;
    } else{
        play_pos = bb.red_pos;
        oppo_pos = bb.blk_pos;
    }

    uint32_t occupied = (play_pos | oppo_pos);
    uint32_t empty = ~occupied;

    uint32_t jump_locations = 0;
    uint32_t jumpers        = 0;
    uint32_t captured       = 0;

    uint32_t pieces_adj_to_opp = BCKWD(turn, oppo_pos) & play_pos & ~bb.king_pos & follow_mask;
    bool has_jumps = false;
    if (pieces_adj_to_opp){
        jump_locations = FORWD(turn, (FORWD(turn, pieces_adj_to_opp) & oppo_pos)) & FORWD_JUMP(turn, pieces_adj_to_opp) & empty;
        if (jump_locations){
            jumpers = BCKWD_JUMP(turn, jump_locations) & pieces_adj_to_opp & BCKWD(turn, (BCKWD(turn, jump_locations) & oppo_pos));
            //if (jumpers){
            //    captured = FORWD(turn, jumpers) & BCKWD(turn, jump_locations) & oppo_pos;
                has_jumps = true;
            //}
        }
    }



    uint32_t king_jump_locations = 0;
    uint32_t king_jumpers        = 0;
    uint32_t captured_by_king    = 0;

    uint32_t kings_adj_to_opp = (FORWD(turn, oppo_pos) | BCKWD(turn, oppo_pos)) & play_pos & bb.king_pos & follow_mask;
    bool has_king_jumps = false;
    if (kings_adj_to_opp){
        king_jump_locations = (FORWD(turn, (FORWD(turn, kings_adj_to_opp) & oppo_pos)) | BCKWD(turn, (BCKWD(turn, kings_adj_to_opp) & oppo_pos))) 
                              & (FORWD_JUMP(turn, kings_adj_to_opp) | BCKWD_JUMP(turn, kings_adj_to_opp)) & empty;
        if (king_jump_locations){
            king_jumpers = (FORWD_JUMP(turn, king_jump_locations) | BCKWD_JUMP(turn, king_jump_locations)) 
                           & kings_adj_to_opp 
                           & (FORWD(turn, (FORWD(turn, king_jump_locations) & oppo_pos))
                                 | BCKWD(turn, (BCKWD(turn, king_jump_locations) & oppo_pos)));
            //if (king_jumpers){
            //    captured_by_king = (FORWD(turn, king_jumpers) | BACKWD(turn, king_jumpers))
            //                           & (FORWD(turn, king_jump_locations) | BCKWD(turn, king_jump_locations))
            //                           & oppo_pos;
                has_king_jumps = true;
            //}
        }
    }

    if (!(has_jumps || has_king_jumps)){
        return children;
    }


    uint32_t jumpers_remaining = jumpers;

    uint32_t king_jumpers_remaining = king_jumpers;


    for (size_t r = 0; r < 8; r++){
        if (ROW_MASK(r) & (jumpers_remaining | king_jumpers_remaining)){
            // JUMPERS ON THIS ROW
            for (size_t c = 0; c < 4; c++){
                uint32_t j_piece = COL_MASK(c) & ROW_MASK(r) & (jumpers_remaining | king_jumpers_remaining);
                if (j_piece){                           // MOVE FOUND: a valid jump can be made from r,c
                    // find individual valid jump locations from here
                    uint32_t j_loc[4] = {0};        // individual jump locations from this piece
                    uint32_t j_cap[4] = {0};        // individual captured piece for the corresponding jump
                    if(c+1 < 4){
                        j_loc[0] = COL_MASK(c+1) & FORWD_JUMP(turn, j_piece) & FORWD(turn, (FORWD(turn, j_piece) & oppo_pos)) & empty;
                        if (j_loc[0]){
                            j_cap[0] = BCKWD(turn, j_loc[0]) & FORWD(turn, j_piece);
                        }
                    }
                    if (c-1 >= 0){
                        j_loc[1] = COL_MASK(c-1) & FORWD_JUMP(turn, j_piece) & FORWD(turn, (FORWD(turn, j_piece) & oppo_pos)) & empty;
                        if (j_loc[1]){
                            j_cap[1] = BCKWD(turn, j_loc[1]) & FORWD(turn, j_piece);
                        }
                    }
                    if(j_piece & king_jumpers_remaining){
                        if(c+1 < 4){
                            j_loc[2] = COL_MASK(c+1) & BCKWD_JUMP(turn, j_piece) & BCKWD(turn, (BCKWD(turn, j_piece) & oppo_pos)) & empty;
                            if (j_loc[2]){
                                j_cap[2] = FORWD(turn, j_loc[2]) & BCKWD(turn, j_piece);
                            }
                        }
                        if (c-1 >= 0){
                            j_loc[3] = COL_MASK(c-1) & BCKWD_JUMP(turn, j_piece) & BCKWD(turn, (BCKWD(turn, j_piece) & oppo_pos)) & empty;
                            if (j_loc[3]){
                                j_cap[3] = BCKWD(turn, j_loc[3]) & FORWD(turn, j_piece);
                            }
                        }
                    }
                    // add new boards to children
                    for(int i = 0; i < 4; i++){
                        if(j_loc[i]){
                            BitBoard child = bb;
                            uint32_t new_play_pos = (play_pos & ~j_piece) | j_loc[i];   // remove j_piece and add it to j_loc[ation]
                            uint32_t new_oppo_pos = oppo_pos & ~j_cap[i];               // remove captured piece from opponent board
                            // find any subsequent jumps resulting from this bit board
                            //      the follow mask is j_loc, the placed we moved to. Only jumps from this piece are valid
                            vector<BitBoard> childs_children = follow_jumps(child,j_loc[i]);
                            if(childs_children.size() == 0){
                                child.turn = !turn;
                                children.push_back(child);
                            } else{
                                for( auto c : childs_children ){
                                    children.push_back(c);
                                }
                            }
                        }
                    }
                }
                if(j_piece & king_jumpers_remaining){
                    king_jumpers_remaining &= ~j_piece;
                } else{
                    jumpers_remaining &= ~j_piece;
                }
                if (!(ROW_MASK(r) & (jumpers_remaining | king_jumpers_remaining))){      // no remaining moves in row
                    break;
                }
            }
        }
    }

    return children;
}

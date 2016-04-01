// Release 0.2
// Adam Levy

#include "checkerboard.hpp"

size_t bit_count(uint32_t i)
{
     i = i - ((i >> 1) & 0x55555555);
     i = (i & 0x33333333) + ((i >> 2) & 0x33333333);
     return (((i + (i >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24;
}

uint32_t get_movers(BitBoard const & bb, bool kings)/*{{{*/
{
    uint32_t play_pos;
    uint32_t oppo_pos;

    if (bb.turn == BLK){
        play_pos = bb.blk_pos;
        oppo_pos = bb.red_pos;
    } else{
        play_pos = bb.red_pos;
        oppo_pos = bb.blk_pos;
    }
    bool turn = bb.turn;

    uint32_t occupied = (play_pos | oppo_pos);
    uint32_t empty = ~occupied;

    uint32_t movers      = BCKWD(turn, empty) & play_pos & ~bb.king_pos;
    uint32_t king_movers = (BCKWD(turn, empty) | FORWD(turn, empty)) & play_pos & bb.king_pos;

    return ((0xffFFffFF * !(kings)) & movers) | ((0xffFFffFF *  (kings)) & king_movers);
}/*}}}*/

uint32_t get_move_locations(const BitBoard &bb, bool kings, uint32_t move_mask)/*{{{*/
{
    uint32_t play_pos;
    uint32_t oppo_pos;

    if (bb.turn == BLK){
        play_pos = bb.blk_pos;
        oppo_pos = bb.red_pos;
    } else{
        play_pos = bb.red_pos;
        oppo_pos = bb.blk_pos;
    }
    bool turn = bb.turn;

    uint32_t occupied = (play_pos | oppo_pos);
    uint32_t empty = ~occupied;

    uint32_t movers      = BCKWD(turn, empty) & play_pos & ~bb.king_pos & move_mask;
    uint32_t move_locations = FORWD(turn, movers) & empty;
    uint32_t king_movers = (BCKWD(turn, empty) | FORWD(turn, empty)) & play_pos & bb.king_pos & move_mask;
    uint32_t king_move_locations = (BCKWD(turn, king_movers) | FORWD(turn, king_movers)) & empty;

    return ((0xffFFffFF * !(kings)) & move_locations) | ((0xffFFffFF *  (kings)) & king_move_locations);
}/*}}}*/

uint32_t get_jumpers(const BitBoard & bb, bool kings)/*{{{*/
{
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
    uint32_t pieces_adj_to_opp = BCKWD(turn, oppo_pos) & play_pos & ~bb.king_pos;
    uint32_t kings_adj_to_opp = (BCKWD(turn, oppo_pos) | FORWD(turn, oppo_pos)) & play_pos & bb.king_pos;

    uint32_t jump_locations = FORWD(turn, (FORWD(turn, pieces_adj_to_opp) & oppo_pos))
                            & FORWD_JUMP(turn, pieces_adj_to_opp)
                            & empty;

    uint32_t jumpers = BCKWD_JUMP(turn, jump_locations)
                     & pieces_adj_to_opp
                     & BCKWD(turn, (BCKWD(turn, jump_locations) & oppo_pos));


    uint32_t king_jump_locations = (FORWD(turn, (FORWD(turn, kings_adj_to_opp) & oppo_pos))
                                    | BCKWD(turn, (BCKWD(turn, kings_adj_to_opp) & oppo_pos)))
                                 & (FORWD_JUMP(turn, kings_adj_to_opp) | BCKWD_JUMP(turn, kings_adj_to_opp))
                                 & empty;

    uint32_t king_jumpers = (FORWD_JUMP(turn, king_jump_locations)
                             | BCKWD_JUMP(turn, king_jump_locations))
                          & kings_adj_to_opp
                          & (FORWD(turn, (FORWD(turn, king_jump_locations) & oppo_pos))
                             | BCKWD(turn, (BCKWD(turn, king_jump_locations) & oppo_pos)));

    return ((0xffFFffFF * !(kings)) & jumpers) | ((0xffFFffFF *  (kings)) & king_jumpers);
}/*}}}*/

uint32_t get_jump_locations(const BitBoard &bb, bool kings, uint32_t move_mask)/*{{{*/
{
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
    uint32_t pieces_adj_to_opp = BCKWD(turn, oppo_pos) & play_pos & ~bb.king_pos & move_mask;
    uint32_t kings_adj_to_opp = (BCKWD(turn, oppo_pos) | FORWD(turn, oppo_pos)) & play_pos & bb.king_pos & move_mask;

    uint32_t jump_locations = FORWD(turn, (FORWD(turn, pieces_adj_to_opp) & oppo_pos))
                            & FORWD_JUMP(turn, pieces_adj_to_opp)
                            & empty;

    uint32_t king_jump_locations = (FORWD(turn, (FORWD(turn, kings_adj_to_opp) & oppo_pos))
                                    | BCKWD(turn, (BCKWD(turn, kings_adj_to_opp) & oppo_pos)))
                                 & (FORWD_JUMP(turn, kings_adj_to_opp) | BCKWD_JUMP(turn, kings_adj_to_opp))
                                 & empty;

    return ((0xffFFffFF * !(kings)) & jump_locations) | ((0xffFFffFF *  (kings)) & king_jump_locations);
}/*}}}*/

vector<BitBoard> gen_children(const BitBoard & bb)/*{{{*/
{
    vector<BitBoard> children = follow_jumps(bb);

    if (children.size() > 0){
        return children;
    }

    uint32_t play_pos;
    uint32_t oppo_pos;

    if (bb.turn == BLK){
        play_pos = bb.blk_pos;
        oppo_pos = bb.red_pos;
    } else{
        play_pos = bb.red_pos;
        oppo_pos = bb.blk_pos;
    }
    bool turn = bb.turn;

    uint32_t occupied = (play_pos | oppo_pos);
    uint32_t empty = ~occupied;

    uint32_t movers      = BCKWD(turn, empty) & play_pos & ~bb.king_pos;
    uint32_t king_movers = (BCKWD(turn, empty) | FORWD(turn, empty)) & play_pos & bb.king_pos;

    // generate child boards
    uint32_t movers_remaining      = movers;
    uint32_t king_movers_remaining = king_movers;
    for (size_t r = 0; r < 8; r++){
        if (ROW_MASK(r) & (movers_remaining | king_movers_remaining)){
            // MOVES ON THIS ROW
            for (size_t c = 0; c < 4; c++){
                uint32_t p_piece = COL_MASK(c) & ROW_MASK(r) & (movers_remaining | king_movers_remaining);
                if (p_piece){                           // MOVE FOUND: a valid move can be made from a piece on r,c
                    // find individual valid move locations from here
                    uint32_t p_loc[4] = {0};        // individual move locations from this piece
                    p_loc[0] = FORWD_4(turn, p_piece) & empty;
                    p_loc[1] = FORWD(turn, p_piece) & empty & ~p_loc[0];
                    bool is_king = p_piece & king_movers_remaining;
                    if (is_king){
                        p_loc[2] = BCKWD_4(turn, p_piece) & empty;
                        p_loc[3] = BCKWD(turn, p_piece) & empty & ~p_loc[2];
                    }
                    // add new boards to children
                    for (size_t i = 0; i < 4; i++){
                        if (p_loc[i]){
                            BitBoard child = bb;
                            uint32_t new_play_pos = (play_pos & ~p_piece) | p_loc[i];   // remove p_piece and add it to p_loc[ation]
                            if (turn == BLK){
                                child.blk_pos = new_play_pos;
                            } else{
                                child.red_pos = new_play_pos;
                            }
                            if(is_king){
                                child.king_pos &= ~p_piece;
                                child.king_pos |= p_loc[i];
                            }
                            if (!is_king && (KING_ME_ROW_MASK(turn) & p_loc[i])){                     // check for king_me
                                child.king_pos |= p_loc[i];
                            }
                            child.turn = !turn;

                            children.push_back(child);
                        }
                    }
                    if(is_king){
                        king_movers_remaining &= ~p_piece;
                    } else{
                        movers_remaining &= ~p_piece;
                    }
                }
                // uncheck move location from remaining moves
                if (!(ROW_MASK(r) & (movers_remaining | king_movers_remaining))){      // no remaining moves in row
                    break;
                }
            }
        }
    }

    return children;
}/*}}}*/

vector<BitBoard> follow_jumps(const BitBoard & bb, uint32_t follow_mask)/*{{{*/
{
    vector<BitBoard> children;

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
    uint32_t pieces_adj_to_opp = BCKWD(turn, oppo_pos) & play_pos & ~bb.king_pos & follow_mask;
    uint32_t kings_adj_to_opp = (BCKWD(turn, oppo_pos) | FORWD(turn, oppo_pos)) & play_pos & bb.king_pos & follow_mask;

    uint32_t jump_locations = FORWD(turn, (FORWD(turn, pieces_adj_to_opp) & oppo_pos))
                            & FORWD_JUMP(turn, pieces_adj_to_opp)
                            & empty;

    uint32_t jumpers = BCKWD_JUMP(turn, jump_locations)
                     & pieces_adj_to_opp
                     & BCKWD(turn, (BCKWD(turn, jump_locations) & oppo_pos));


    uint32_t king_jump_locations = (FORWD(turn, (FORWD(turn, kings_adj_to_opp) & oppo_pos))
                                    | BCKWD(turn, (BCKWD(turn, kings_adj_to_opp) & oppo_pos)))
                                 & (FORWD_JUMP(turn, kings_adj_to_opp) | BCKWD_JUMP(turn, kings_adj_to_opp))
                                 & empty;

    uint32_t king_jumpers = (FORWD_JUMP(turn, king_jump_locations)
                             | BCKWD_JUMP(turn, king_jump_locations))
                          & kings_adj_to_opp
                          & (FORWD(turn, (FORWD(turn, king_jump_locations) & oppo_pos))
                             | BCKWD(turn, (BCKWD(turn, king_jump_locations) & oppo_pos)));

    if (!(jumpers | king_jumpers)){
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
                    if (c < 3){
                        j_loc[0] = COL_MASK(c+1) & FORWD_JUMP(turn, j_piece) & FORWD(turn, (FORWD(turn, j_piece) & oppo_pos)) & empty;
                        if (j_loc[0]){
                            j_cap[0] = BCKWD(turn, j_loc[0]) & FORWD(turn, j_piece);
                        }
                    }
                    if (c > 0){
                        j_loc[1] = COL_MASK(c-1) & FORWD_JUMP(turn, j_piece) & FORWD(turn, (FORWD(turn, j_piece) & oppo_pos)) & empty;
                        if (j_loc[1]){
                            j_cap[1] = BCKWD(turn, j_loc[1]) & FORWD(turn, j_piece);
                        }
                    }
                    bool is_king = j_piece & king_jumpers_remaining;
                    if (is_king){
                        if (c < 3){
                            j_loc[2] = COL_MASK(c+1) & BCKWD_JUMP(turn, j_piece) & BCKWD(turn, (BCKWD(turn, j_piece) & oppo_pos)) & empty;
                            if (j_loc[2]){
                                j_cap[2] = FORWD(turn, j_loc[2]) & BCKWD(turn, j_piece);
                            }
                        }
                        if (c > 0){
                            j_loc[3] = COL_MASK(c-1) & BCKWD_JUMP(turn, j_piece) & BCKWD(turn, (BCKWD(turn, j_piece) & oppo_pos)) & empty;
                            if (j_loc[3]){
                                j_cap[3] = FORWD(turn, j_loc[3]) & BCKWD(turn, j_piece);
                            }
                        }
                    }
                    // add new boards to children
                    for (size_t i = 0; i < 4; i++){
                        if (j_loc[i]){
                            BitBoard child = bb;
                            uint32_t new_play_pos = (play_pos & ~j_piece) | j_loc[i];   // remove j_piece and add it to j_loc[ation]
                            uint32_t new_oppo_pos = oppo_pos & ~j_cap[i];               // remove captured piece from opponent board
                            child.king_pos &= ~j_cap[i];
                            if (turn == BLK){
                                child.blk_pos = new_play_pos;
                                child.red_pos = new_oppo_pos;
                            } else{
                                child.blk_pos = new_oppo_pos;
                                child.red_pos = new_play_pos;
                            }
                            if(is_king){
                                child.king_pos &= ~j_piece;
                                child.king_pos |= j_loc[i];
                            }
                            bool kinged = false;
                            if (!is_king && (KING_ME_ROW_MASK(turn) & j_loc[i])){                     // check for king_me
                                child.king_pos |= j_loc[i];
                                kinged = true;
                            }

                            vector<BitBoard> childs_children;
                            if(!kinged && new_oppo_pos){  // did not just get kinged && opponent still has pieces
                                // find any subsequent jumps resulting from this bit board
                                //      the follow mask is j_loc, the placed we moved to. Only jumps from this piece are valid
                                childs_children = follow_jumps(child, j_loc[i]);
                            }
                            if (childs_children.size() == 0){
                                child.turn = !turn;
                                children.push_back(child);
                            } else{
                                for ( auto c : childs_children ){
                                    if (childs_children.size() > 1){
                                        bool repeat = false;
                                        for (auto cc : children){
                                            if (c == cc){
                                                repeat = true;
                                                break;
                                            }
                                        }
                                        if (repeat){
                                            continue;
                                        }
                                    }
                                    children.push_back(c);
                                }
                            }
                        }
                    }
                    if (is_king){
                        king_jumpers_remaining &= ~j_piece;
                    } else{
                        jumpers_remaining &= ~j_piece;
                    }
                }
                if (!(ROW_MASK(r) & (jumpers_remaining | king_jumpers_remaining))){      // no remaining moves in row
                    break;
                }
            }
        }
    }

    return children;
}/*}}}*/

CheckerBoard::CheckerBoard()/*{{{*/
{}/*}}}*/

CheckerBoard::CheckerBoard(BitBoard bb) : m_bb(bb){};

bool CheckerBoard::is_occupied( size_t pos_index )/*{{{*/
{
    return POS_MASK[pos_index] & (m_bb.red_pos | m_bb.blk_pos);
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
    cout << " ----------------------------- " << endl;
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
            cout << "  " << line << endl;
            line.clear();
        }
    }
    cout << " ----------------------------- " << endl;
    cout << endl;

}/*}}}*/

void CheckerBoard::gen_children()/*{{{*/
{
    m_children = follow_jumps(m_bb);

    if (m_children.size() > 0){
        return;
    }

    uint32_t play_pos;
    uint32_t oppo_pos;

    if (m_bb.turn == BLK){
        play_pos = m_bb.blk_pos;
        oppo_pos = m_bb.red_pos;
    } else{
        play_pos = m_bb.red_pos;
        oppo_pos = m_bb.blk_pos;
    }
    bool turn = m_bb.turn;

    uint32_t occupied = (play_pos | oppo_pos);
    uint32_t empty = ~occupied;

    uint32_t movers      = BCKWD(turn, empty) & play_pos & ~m_bb.king_pos;
    uint32_t king_movers = (BCKWD(turn, empty) | FORWD(turn, empty)) & play_pos & m_bb.king_pos;

    // generate child boards
    uint32_t movers_remaining      = movers;
    uint32_t king_movers_remaining = king_movers;
    for (size_t r = 0; r < 8; r++){
        if (ROW_MASK(r) & (movers_remaining | king_movers_remaining)){
            // MOVES ON THIS ROW
            for (size_t c = 0; c < 4; c++){
                uint32_t p_piece = COL_MASK(c) & ROW_MASK(r) & (movers_remaining | king_movers_remaining);
                if (p_piece){                           // MOVE FOUND: a valid move can be made from a piece on r,c
                    // find individual valid move locations from here
                    uint32_t p_loc[4] = {0};        // individual move locations from this piece
                    p_loc[0] = FORWD_4(turn, p_piece) & empty;
                    p_loc[1] = FORWD(turn, p_piece) & empty & ~p_loc[0];
                    bool is_king = p_piece & king_movers_remaining;
                    if (is_king){
                        p_loc[2] = BCKWD_4(turn, p_piece) & empty;
                        p_loc[3] = BCKWD(turn, p_piece) & empty & ~p_loc[2];
                    }
                    // add new boards to children
                    for (size_t i = 0; i < 4; i++){
                        if (p_loc[i]){
                            BitBoard child = m_bb;
                            uint32_t new_play_pos = (play_pos & ~p_piece) | p_loc[i];   // remove p_piece and add it to p_loc[ation]
                            if (turn == BLK){
                                child.blk_pos = new_play_pos;
                            } else{
                                child.red_pos = new_play_pos;
                            }
                            if(is_king){
                                child.king_pos &= ~p_piece;
                                child.king_pos |= p_loc[i];
                            }
                            if (!is_king && (KING_ME_ROW_MASK(turn) & p_loc[i])){                     // check for king_me
                                child.king_pos |= p_loc[i];
                            }
                            child.turn = !turn;

                            m_children.push_back(child);
                        }
                    }
                    if(is_king){
                        king_movers_remaining &= ~p_piece;
                    } else{
                        movers_remaining &= ~p_piece;
                    }
                }
                // uncheck move location from remaining moves
                if (!(ROW_MASK(r) & (movers_remaining | king_movers_remaining))){      // no remaining moves in row
                    break;
                }
            }
        }
    }

}/*}}}*/


bool BitBoard::operator==(const BitBoard &rhs) const {/*{{{*/
    return (red_pos == rhs.red_pos &&
            blk_pos == rhs.blk_pos &&
            king_pos == rhs.king_pos &&
            turn == rhs.turn);
}/*}}}*/

vector<BitBoard> CheckerBoard::follow_jumps(const BitBoard & bb, uint32_t follow_mask)/*{{{*/
{
    vector<BitBoard> children;

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
    uint32_t pieces_adj_to_opp = BCKWD(turn, oppo_pos) & play_pos & ~bb.king_pos & follow_mask;
    uint32_t kings_adj_to_opp = (BCKWD(turn, oppo_pos) | FORWD(turn, oppo_pos)) & play_pos & bb.king_pos & follow_mask;

    uint32_t jump_locations = FORWD(turn, (FORWD(turn, pieces_adj_to_opp) & oppo_pos))
                            & FORWD_JUMP(turn, pieces_adj_to_opp)
                            & empty;

    uint32_t jumpers = BCKWD_JUMP(turn, jump_locations)
                     & pieces_adj_to_opp
                     & BCKWD(turn, (BCKWD(turn, jump_locations) & oppo_pos));


    uint32_t king_jump_locations = (FORWD(turn, (FORWD(turn, kings_adj_to_opp) & oppo_pos))
                                    | BCKWD(turn, (BCKWD(turn, kings_adj_to_opp) & oppo_pos)))
                                 & (FORWD_JUMP(turn, kings_adj_to_opp) | BCKWD_JUMP(turn, kings_adj_to_opp))
                                 & empty;

    uint32_t king_jumpers = (FORWD_JUMP(turn, king_jump_locations)
                             | BCKWD_JUMP(turn, king_jump_locations))
                          & kings_adj_to_opp
                          & (FORWD(turn, (FORWD(turn, king_jump_locations) & oppo_pos))
                             | BCKWD(turn, (BCKWD(turn, king_jump_locations) & oppo_pos)));

    if (!(jumpers | king_jumpers)){
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
                    if (c < 3){
                        j_loc[0] = COL_MASK(c+1) & FORWD_JUMP(turn, j_piece) & FORWD(turn, (FORWD(turn, j_piece) & oppo_pos)) & empty;
                        if (j_loc[0]){
                            j_cap[0] = BCKWD(turn, j_loc[0]) & FORWD(turn, j_piece);
                        }
                    }
                    if (c > 0){
                        j_loc[1] = COL_MASK(c-1) & FORWD_JUMP(turn, j_piece) & FORWD(turn, (FORWD(turn, j_piece) & oppo_pos)) & empty;
                        if (j_loc[1]){
                            j_cap[1] = BCKWD(turn, j_loc[1]) & FORWD(turn, j_piece);
                        }
                    }
                    bool is_king = j_piece & king_jumpers_remaining;
                    if (is_king){
                        if (c < 3){
                            j_loc[2] = COL_MASK(c+1) & BCKWD_JUMP(turn, j_piece) & BCKWD(turn, (BCKWD(turn, j_piece) & oppo_pos)) & empty;
                            if (j_loc[2]){
                                j_cap[2] = FORWD(turn, j_loc[2]) & BCKWD(turn, j_piece);
                            }
                        }
                        if (c > 0){
                            j_loc[3] = COL_MASK(c-1) & BCKWD_JUMP(turn, j_piece) & BCKWD(turn, (BCKWD(turn, j_piece) & oppo_pos)) & empty;
                            if (j_loc[3]){
                                j_cap[3] = FORWD(turn, j_loc[3]) & BCKWD(turn, j_piece);
                            }
                        }
                    }
                    // add new boards to children
                    for (size_t i = 0; i < 4; i++){
                        if (j_loc[i]){
                            BitBoard child = bb;
                            uint32_t new_play_pos = (play_pos & ~j_piece) | j_loc[i];   // remove j_piece and add it to j_loc[ation]
                            uint32_t new_oppo_pos = oppo_pos & ~j_cap[i];               // remove captured piece from opponent board
                            child.king_pos &= ~j_cap[i];
                            if (turn == BLK){
                                child.blk_pos = new_play_pos;
                                child.red_pos = new_oppo_pos;
                            } else{
                                child.blk_pos = new_oppo_pos;
                                child.red_pos = new_play_pos;
                            }
                            if(is_king){
                                child.king_pos &= ~j_piece;
                                child.king_pos |= j_loc[i];
                            }
                            bool kinged = false;
                            if (!is_king && (KING_ME_ROW_MASK(turn) & j_loc[i])){                     // check for king_me
                                child.king_pos |= j_loc[i];
                                kinged = true;
                            }
                            
                            vector<BitBoard> childs_children;
                            if(!kinged && new_oppo_pos){  // did not just get kinged && opponent still has pieces
                                // find any subsequent jumps resulting from this bit board
                                //      the follow mask is j_loc, the placed we moved to. Only jumps from this piece are valid
                                childs_children = follow_jumps(child, j_loc[i]);
                            }
                            if (childs_children.size() == 0){
                                child.turn = !turn;
                                children.push_back(child);
                            } else{
                                for ( auto c : childs_children ){
                                    if (childs_children.size() > 1){
                                        bool repeat = false;
                                        for (auto cc : children){
                                            if (c == cc){
                                                repeat = true;
                                                break;
                                            }
                                        }
                                        if (repeat){
                                            continue;
                                        }
                                    }
                                    children.push_back(c);
                                }
                            }
                        }
                    }
                    if (is_king){
                        king_jumpers_remaining &= ~j_piece;
                    } else{
                        jumpers_remaining &= ~j_piece;
                    }
                }
                if (!(ROW_MASK(r) & (jumpers_remaining | king_jumpers_remaining))){      // no remaining moves in row
                    break;
                }
            }
        }
    }

    return children;
}/*}}}*/

string to_string( const BitBoard & bb )/*{{{*/
{
    string s_board;
    s_board.resize(NUM_POS);

    for (int r = 7; r >= 0; r--){
        for (size_t c = 0; c < 4; c++){
            size_t i = r * 4 + c;
            bool has_piece = POS_MASK[i] & (bb.red_pos | bb.blk_pos);
            string sym = "_";

            if (has_piece){
                bool is_red = POS_MASK[i] & bb.red_pos;
                bool is_king = POS_MASK[i] & bb.king_pos;
                if (is_red){
                    sym = "r";
                } else{
                    sym = "b";
                }
                if (is_king){
                    sym[0] += 'A' - 'a';
                } 
            }

            s_board += sym;
        }
    }

    return s_board;
}/*}}}*/

BitBoard from_string( const string & s_board, bool turn )/*{{{*/
{
    BitBoard bb;
    bb.red_pos = 0;
    bb.blk_pos = 0;
    bb.king_pos = 0;
    bb.turn = turn;

    if(s_board.size() != NUM_POS){
        cout << "ERROR STRING TOO SHORT" << endl;
        return bb;
    }


    size_t r = 7;
    size_t c = 0;
    for(size_t i = 0; i < NUM_POS; i++){
        size_t p = r * 4 + c;

        char sym = s_board[i];

        if (sym != '_'){
            if (sym == 'r' || sym == 'R'){
                bb.red_pos |= POS_MASK[p];
            } else{
                bb.blk_pos |= POS_MASK[p];
            }
            if (sym == 'R' || sym == 'B'){
                bb.king_pos |= POS_MASK[p];
            }
        }
        c++;
        if (c == 4){
            c = 0;
            r--;
        }
    }
    return bb;
}/*}}}*/




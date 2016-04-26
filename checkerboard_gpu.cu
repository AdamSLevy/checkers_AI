// Release 0.2
// Adam Levy

#include "checkerboard_gpu.hpp"

__device__ bool BitBoard_gpu::operator==(const BitBoard_gpu &rhs) const {/*{{{*/
    return ( red_pos == rhs.red_pos  &&
             blk_pos == rhs.blk_pos  &&
            king_pos == rhs.king_pos &&
                turn == rhs.turn);
}/*}}}*/

__device__ bool BitBoard_gpu::operator!=(const BitBoard_gpu &rhs) const {/*{{{*/
    return ( red_pos != rhs.red_pos  ||
             blk_pos != rhs.blk_pos  ||
            king_pos != rhs.king_pos ||
                turn != rhs.turn);
}/*}}}*/

__device__ __host__ BitBoard_gpu& BitBoard_gpu::operator=(const BitBoard &rhs){
    red_pos = rhs.red_pos;
    blk_pos = rhs.blk_pos;
    king_pos = rhs.king_pos;
    turn = rhs.turn;
    return *this;
}

__device__ void push_back(BitBoardArray & bba, const BitBoard_gpu & b)
{
    if (bba.size < BB_PRE_ALLOC){
        bba.bb_ary[bba.size] = b;
        bba.size++;
    } else{
        printf("Not enough space for moves!\n");
        asm("trap;");
    }
}

__device__ BitBoardArray gen_children_gpu(const BitBoard_gpu & bb)/*{{{*/
{
    BitBoardArray children = follow_jumps_gpu(bb);

    if (children.size > 0){
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

    if (!(movers | king_movers)){
        return children;
    }

    children.bb_ary = new BitBoard_gpu[BB_PRE_ALLOC];

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
                            BitBoard_gpu child = bb;
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

                            push_back(children, child);
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

    if (children.size == 0){
        delete [] children.bb_ary;
    }
    return children;
}/*}}}*/

__device__ BitBoardArray follow_jumps_gpu(const BitBoard_gpu & bb, uint32_t follow_mask)/*{{{*/
{
    BitBoardArray children;

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

    children.bb_ary = new BitBoard_gpu[BB_PRE_ALLOC];

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
                            BitBoard_gpu child = bb;
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

                            BitBoardArray childs_children;
                            if(!kinged && new_oppo_pos){  // did not just get kinged && opponent still has pieces
                                // find any subsequent jumps resulting from this bit board
                                //      the follow mask is j_loc, the placed we moved to. Only jumps from this piece are valid
                                childs_children = follow_jumps_gpu(child, j_loc[i]);
                            }
                            if (childs_children.size == 0){
                                child.turn = !turn;
                                push_back(children, child);
                            } else{
                                for ( int i = 0; i < childs_children.size; i++ ){
                                    auto c = childs_children.bb_ary[i];
                                    if (childs_children.size > 1){
                                        bool repeat = false;
                                        for (int j = 0; j < children.size; j++){
                                            auto cc = children.bb_ary[j];
                                            if (c == cc){
                                                repeat = true;
                                                break;
                                            }
                                        }
                                        if (repeat){
                                            continue;
                                        }
                                    }
                                    push_back(children, c);
                                }
                                delete [] childs_children.bb_ary;
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

    if (children.size == 0){
        delete [] children.bb_ary;
    }
    return children;
}/*}}}*/

__device__ size_t bit_count_gpu(uint32_t i)
{
     i = i - ((i >> 1) & 0x55555555);
     i = (i & 0x33333333) + ((i >> 2) & 0x33333333);
     return (((i + (i >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24;
}


#include "static_eval.hpp"

int static_eval(BitBoard bb, bool player)
{
    uint32_t red_piece = bb.red_pos & ~bb.king_pos;
    uint32_t red_king  = bb.red_pos &  bb.king_pos;
    uint32_t blk_piece = bb.blk_pos & ~bb.king_pos;
    uint32_t blk_king  = bb.blk_pos &  bb.king_pos;

    int red_score =  bit_count(bb.red_pos);
    red_score +=     bit_count(ROW_MASK(4) & red_piece);
    red_score += 2 * bit_count(ROW_MASK(3) & red_piece);
    red_score += 3 * bit_count(ROW_MASK(2) & red_piece);
    red_score += 4 * bit_count(ROW_MASK(1) & red_piece);
    red_score += 5 * bit_count(red_king);
    
    int blk_score =  bit_count(bb.blk_pos);
    blk_score +=     bit_count(ROW_MASK(3) & blk_piece);
    blk_score += 2 * bit_count(ROW_MASK(4) & blk_piece);
    blk_score += 3 * bit_count(ROW_MASK(5) & blk_piece);
    blk_score += 4 * bit_count(ROW_MASK(6) & blk_piece);
    blk_score += 5 * bit_count(blk_king);

    int value = 0;
    if (player == BLK){
        value = blk_score - red_score;
    } else{
        value = red_score - blk_score;
    }
    
    return value;
}

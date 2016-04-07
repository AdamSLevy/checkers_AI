#ifndef BIT_MASK_INIT
#define BIT_MASK_INIT

// Release 0.1
// Adam Levy


//                        00  01  02  03  
//                        ----------------|
//   1C  1D  1E  1F       1C  1D  1E  1F  |   07
// 18  19  1A  1B         18  19  1A  1B  |   06
//   14  15  16  17       14  15  16  17  |   05
// 10  11  12  13         10  11  12  13  |   04
//   0C  0D  0E  0F       0C  0D  0E  0F  |   03
// 08  09  0A  0B         08  09  0A  0B  |   02
//   04  05  06  07       04  05  06  07  |   01
// 00  01  02  03         00  01  02  03  |   00

#include <stdint.h>
#define ROW_MASK(row) (0xf << (4*(row)))
#define COL_MASK(col) (0x11111111 << (col))

// MOVE SHIFT MASKS
#define LEFT3_MASK       ((~COL_MASK(0)) & (ROW_MASK(0) | ROW_MASK(2) | ROW_MASK(4) | ROW_MASK(6)))           //0b00001110000011100000111000001110
#define LEFT5_MASK       ((~COL_MASK(3)) & (ROW_MASK(1) | ROW_MASK(3) | ROW_MASK(5)))                         //0b00000000011100000111000001110000;
#define RIGHT3_MASK      ((~COL_MASK(3)) & (ROW_MASK(1) | ROW_MASK(3) | ROW_MASK(5) | ROW_MASK(7)))           //0b01110000011100000111000001110000;
#define RIGHT5_MASK      ((~COL_MASK(0)) & (ROW_MASK(2) | ROW_MASK(4) | ROW_MASK(6)))                         //0b00000000111000001110000011100000;
// JUMP SHIFT MASKS
#define LEFT7_MASK       ((~COL_MASK(0)) & (~(ROW_MASK(6) | ROW_MASK(7))))
#define LEFT9_MASK       ((~COL_MASK(3)) & (~(ROW_MASK(6) | ROW_MASK(7))))
#define RIGHT7_MASK      ((~COL_MASK(3)) & (~(ROW_MASK(0) | ROW_MASK(1))))
#define RIGHT9_MASK      ((~COL_MASK(0)) & (~(ROW_MASK(0) | ROW_MASK(1))))

// GENERATE MOVES OR PIECES CONCURRENTLY ON A BOARD
#define MOVE_LEFT(pos)      (((pos) << 4) | (((pos) &  LEFT3_MASK) << 3) | (((pos) &  LEFT5_MASK) << 5))
#define MOVE_RIGHT(pos)     (((pos) >> 4) | (((pos) & RIGHT3_MASK) >> 3) | (((pos) & RIGHT5_MASK) >> 5))

#define LEFT_4(pos)         ((pos) << 4)
#define RIGHT_4(pos)        ((pos) >> 4)

// GENERATE JUMP POSITIONS OR JUMP PIECES CONCURRENTLY ON A BOARD
#define JUMP_LEFT(pos)      ((((pos) &  LEFT7_MASK) << 7) | (((pos) &  LEFT9_MASK) << 9))
#define JUMP_RIGHT(pos)     ((((pos) & RIGHT7_MASK) >> 7) | (((pos) & RIGHT9_MASK) >> 9))

// FORWARD GIVES VALID POTENTIAL MOVES FROM THE GIVEN PIECES (pos)
#define BLK_FORWD(pos)      MOVE_LEFT(pos)
#define BLK_FORWD_4(pos)    LEFT_4(pos)
#define BLK_JUMP(pos)       JUMP_LEFT(pos)
// BACKWARD GIVES PIECES THAT COULD HAVE MOVED FROM THE GIVEN POSITIONS (pos)
#define BLK_BCKWD(pos)      MOVE_RIGHT(pos)
#define BLK_BCKWD_4(pos)    RIGHT_4(pos)
#define BLK_JUMP_BACK(pos)  JUMP_RIGHT(pos)

// SAME FOR RED
#define RED_FORWD(pos)      MOVE_RIGHT(pos)
#define RED_FORWD_4(pos)    RIGHT_4(pos)
#define RED_JUMP(pos)       JUMP_RIGHT(pos)
#define RED_BCKWD(pos)      MOVE_LEFT(pos)
#define RED_BCKWD_4(pos)    LEFT_4(pos)
#define RED_JUMP_BACK(pos)  JUMP_LEFT(pos)

#define IS_RED(turn)        (0xffFFffFF * !(turn))
#define IS_BLK(turn)        (0xffFFffFF *  (turn))

// REMOVE THE GUESS WORK AND SAVE REDUNDANT CODE
#define FORWD(turn, pos)        ((BLK_FORWD(pos)        & IS_BLK(turn)) | (RED_FORWD(pos)      & IS_RED(turn)))
#define FORWD_4(turn, pos)      ((BLK_FORWD_4(pos)      & IS_BLK(turn)) | (RED_FORWD_4(pos)    & IS_RED(turn)))
#define FORWD_JUMP(turn, pos)   ((BLK_JUMP(pos)         & IS_BLK(turn)) | (RED_JUMP(pos)       & IS_RED(turn)))
                                                                      
#define BCKWD(turn, pos)        ((BLK_BCKWD(pos)        & IS_BLK(turn)) | (RED_BCKWD(pos)      & IS_RED(turn)))
#define BCKWD_4(turn, pos)      ((BLK_BCKWD_4(pos)      & IS_BLK(turn)) | (RED_BCKWD_4(pos)    & IS_RED(turn)))
#define BCKWD_JUMP(turn, pos)   ((BLK_JUMP_BACK(pos)    & IS_BLK(turn)) | (RED_JUMP_BACK(pos)  & IS_RED(turn)))

#define KING_ME_ROW_MASK(turn)  ((ROW_MASK(7)           & IS_BLK(turn)) | (ROW_MASK(0)         & IS_RED(turn)))

#define RED 0
#define BLK 1

#define FIRST_TURN BLK

#define NUM_PIECES      12
#define NUM_POS         0x20
#define LAST_POS_INDEX  (NUM_POS - 1)



//
//
// id: 00,  moves:   black: 04        red:          black_bm: 0b00000000000000000000000000110000  red_bm: 0b00000000000000000000000000000000
// id: 01,  moves:   black: 04,05     red:          black_bm: 0b00000000000000000000000001100000  red_bm: 0b00000000000000000000000000000000
// id: 02,  moves:   black: 05,06     red:          black_bm: 0b00000000000000000000000011000000  red_bm: 0b00000000000000000000000000000000
// id: 03,  moves:   black: 06,07     red:          black_bm: 0b00000000000000000000000010000000  red_bm: 0b00000000000000000000000000000000
//                                    
// id: 04,  moves:   black: 08,09     red:  00,01   black_bm: 0b00000000000000000000000100000000  red_bm: 0b00000000000000000000000000000001
// id: 05,  moves:   black: 09,0A     red:  01,02   black_bm: 0b00000000000000000000001100000000  red_bm: 0b00000000000000000000000000000011
// id: 06,  moves:   black: 0A,0B     red:  02,03   black_bm: 0b00000000000000000000011000000000  red_bm: 0b00000000000000000000000000000110
// id: 07,  moves:   black: 0B        red:  03      black_bm: 0b00000000000000000000110000000000  red_bm: 0b00000000000000000000000000001100
//                                    
// id: 08,  moves:   black: 0C        red:  04      black_bm: 0b00000000000000000011000000000000  red_bm: 0b00000000000000000000000000110000
// id: 09,  moves:   black: 0C,0D     red:  04,05   black_bm: 0b00000000000000000110000000000000  red_bm: 0b00000000000000000000000001100000
// id: 0A,  moves:   black: 0D,0E     red:  05,06   black_bm: 0b00000000000000001100000000000000  red_bm: 0b00000000000000000000000011000000
// id: 0B,  moves:   black: 0E,0F     red:  06,07   black_bm: 0b00000000000000001000000000000000  red_bm: 0b00000000000000000000000010000000
//                                               
// id: 0C,  moves:   black: 10,11     red:  08,09   black_bm: 0b00000000000000010000000000000000  red_bm: 0b00000000000000000000000100000000
// id: 0D,  moves:   black: 11,12     red:  09,0A   black_bm: 0b00000000000000110000000000000000  red_bm: 0b00000000000000000000001100000000
// id: 0E,  moves:   black: 12,13     red:  0A,0B   black_bm: 0b00000000000001100000000000000000  red_bm: 0b00000000000000000000011000000000
// id: 0F,  moves:   black: 13        red:  0B      black_bm: 0b00000000000011000000000000000000  red_bm: 0b00000000000000000000110000000000
//                                               
// id: 10,  moves:   black: 14        red:  0C      black_bm: 0b00000000001100000000000000000000  red_bm: 0b00000000000000000011000000000000
// id: 11,  moves:   black: 14,15     red:  0C,0D   black_bm: 0b00000000011000000000000000000000  red_bm: 0b00000000000000000110000000000000
// id: 12,  moves:   black: 15,16     red:  0D,0E   black_bm: 0b00000000110000000000000000000000  red_bm: 0b00000000000000001100000000000000
// id: 13,  moves:   black: 16,17     red:  0E,0F   black_bm: 0b00000000100000000000000000000000  red_bm: 0b00000000000000001000000000000000
//                                               
// id: 14,  moves:   black: 18,19     red:  10,11   black_bm: 0b00000001000000000000000000000000  red_bm: 0b00000000000000010000000000000000
// id: 15,  moves:   black: 19,1A     red:  11,12   black_bm: 0b00000011000000000000000000000000  red_bm: 0b00000000000000110000000000000000
// id: 16,  moves:   black: 1A,1B     red:  12,13   black_bm: 0b00000110000000000000000000000000  red_bm: 0b00000000000001100000000000000000
// id: 17,  moves:   black: 1B        red:  13      black_bm: 0b00001100000000000000000000000000  red_bm: 0b00000000000011000000000000000000
//                                               
// id: 18,  moves:   black: 1C        red:  14      black_bm: 0b00110000000000000000000000000000  red_bm: 0b00000000001100000000000000000000
// id: 19,  moves:   black: 1C,1D     red:  14,15   black_bm: 0b01100000000000000000000000000000  red_bm: 0b00000000011000000000000000000000
// id: 1A,  moves:   black: 1D,1E     red:  15,16   black_bm: 0b11000000000000000000000000000000  red_bm: 0b00000000110000000000000000000000
// id: 1B,  moves:   black: 1E,1F     red:  16,17   black_bm: 0b10000000000000000000000000000000  red_bm: 0b00000000100000000000000000000000
//                                               
// id: 1C,  moves:   black:           red:  18,19   black_bm: 0b00000000000000000000000000000000  red_bm: 0b00000001000000000000000000000000
// id: 1D,  moves:   black:           red:  19,1A   black_bm: 0b00000000000000000000000000000000  red_bm: 0b00000011000000000000000000000000
// id: 1E,  moves:   black:           red:  1A,1B   black_bm: 0b00000000000000000000000000000000  red_bm: 0b00000110000000000000000000000000
// id: 1F,  moves:   black:           red:  1B      black_bm: 0b00000000000000000000000000000000  red_bm: 0b00001100000000000000000000000000
//



const uint32_t BLK_MOV_MASK[NUM_POS] = { 0b00000000000000000000000000110000,
                                         0b00000000000000000000000001100000,
                                         0b00000000000000000000000011000000,
                                         0b00000000000000000000000010000000,
                                                                           
                                         0b00000000000000000000000100000000,
                                         0b00000000000000000000001100000000,
                                         0b00000000000000000000011000000000,
                                         0b00000000000000000000110000000000,
                                                                           
                                         0b00000000000000000011000000000000,
                                         0b00000000000000000110000000000000,
                                         0b00000000000000001100000000000000,
                                         0b00000000000000001000000000000000,
                                                                           
                                         0b00000000000000010000000000000000,
                                         0b00000000000000110000000000000000,
                                         0b00000000000001100000000000000000,
                                         0b00000000000011000000000000000000,
                                                                           
                                         0b00000000001100000000000000000000,
                                         0b00000000011000000000000000000000,
                                         0b00000000110000000000000000000000,
                                         0b00000000100000000000000000000000,
                                                                           
                                         0b00000001000000000000000000000000,
                                         0b00000011000000000000000000000000,
                                         0b00000110000000000000000000000000,
                                         0b00001100000000000000000000000000,
                                                                           
                                         0b00110000000000000000000000000000,
                                         0b01100000000000000000000000000000,
                                         0b11000000000000000000000000000000,
                                         0b10000000000000000000000000000000,
                                                                           
                                         0b00000000000000000000000000000000,
                                         0b00000000000000000000000000000000,
                                         0b00000000000000000000000000000000,
                                         0b00000000000000000000000000000000 };

const uint32_t RED_MOV_MASK[NUM_POS] = { 0b00000000000000000000000000000000,
                                         0b00000000000000000000000000000000,
                                         0b00000000000000000000000000000000,
                                         0b00000000000000000000000000000000,
                                                                           
                                         0b00000000000000000000000000000001,
                                         0b00000000000000000000000000000011,
                                         0b00000000000000000000000000000110,
                                         0b00000000000000000000000000001100,
                                                                           
                                         0b00000000000000000000000000110000,
                                         0b00000000000000000000000001100000,
                                         0b00000000000000000000000011000000,
                                         0b00000000000000000000000010000000,
                                                                           
                                         0b00000000000000000000000100000000,
                                         0b00000000000000000000001100000000,
                                         0b00000000000000000000011000000000,
                                         0b00000000000000000000110000000000,
                                                                           
                                         0b00000000000000000011000000000000,
                                         0b00000000000000000110000000000000,
                                         0b00000000000000001100000000000000,
                                         0b00000000000000001000000000000000,
                                                                           
                                         0b00000000000000010000000000000000,
                                         0b00000000000000110000000000000000,
                                         0b00000000000001100000000000000000,
                                         0b00000000000011000000000000000000,
                                                                           
                                         0b00000000001100000000000000000000,
                                         0b00000000011000000000000000000000,
                                         0b00000000110000000000000000000000,
                                         0b00000000100000000000000000000000,
                                                                           
                                         0b00000001000000000000000000000000,
                                         0b00000011000000000000000000000000,
                                         0b00000110000000000000000000000000,
                                         0b00001100000000000000000000000000 };

const uint32_t POS_MASK[NUM_POS]     = { 0b00000000000000000000000000000001,
                                         0b00000000000000000000000000000010,
                                         0b00000000000000000000000000000100,
                                         0b00000000000000000000000000001000,

                                         0b00000000000000000000000000010000,
                                         0b00000000000000000000000000100000,
                                         0b00000000000000000000000001000000,
                                         0b00000000000000000000000010000000,

                                         0b00000000000000000000000100000000,
                                         0b00000000000000000000001000000000,
                                         0b00000000000000000000010000000000,
                                         0b00000000000000000000100000000000,

                                         0b00000000000000000001000000000000,
                                         0b00000000000000000010000000000000,
                                         0b00000000000000000100000000000000,
                                         0b00000000000000001000000000000000,

                                         0b00000000000000010000000000000000,
                                         0b00000000000000100000000000000000,
                                         0b00000000000001000000000000000000,
                                         0b00000000000010000000000000000000,

                                         0b00000000000100000000000000000000,
                                         0b00000000001000000000000000000000,
                                         0b00000000010000000000000000000000,
                                         0b00000000100000000000000000000000,

                                         0b00000001000000000000000000000000,
                                         0b00000010000000000000000000000000,
                                         0b00000100000000000000000000000000,
                                         0b00001000000000000000000000000000,

                                         0b00010000000000000000000000000000,
                                         0b00100000000000000000000000000000,
                                         0b01000000000000000000000000000000,
                                         0b10000000000000000000000000000000 };


//const uint32_t RED_INIT_POS_BM  = 0xFFF00000;
//const uint32_t BLK_INIT_POS_BM  = 0x00000FFF;
//const uint32_t KING_INIT_POS_BM = 0x00000000;

#define RED_INIT_POS_BM    0xFFF00000
#define BLK_INIT_POS_BM    0x00000FFF
#define KING_INIT_POS_BM   0x00000000

#endif

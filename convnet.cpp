#include "convnet.h"

void ConvNet::add_input( const BitBoard & bb )
{
    if( bb.turn != BLK ){
        // ROTATE BOARD 180
        PRINT_DEBUG;
        exit(1);
    }

    vector<vector<bool>> all_neurons;

    for (size_t bit_pos = 0; bit_pos < NUM_POS; bit_pos++){
        vector<bool> neuron_input;
        size_t c = bit_pos % 4;
        for (size_t b_r_k = 0; b_r_k < 3; b_r_k++){
            uint32_t board;
            switch (b_r_k){
                case 0:             // BLK
                    board = bb.blk_pos;
                    break;
                case 1:             // RED
                    board = bb.red_pos;
                    break;
                case 2:             // KING
                    board = bb.king_pos;
                    break;
                default:            // NEVER WILL OCCUR. DUMB PROGRAMMER PROTECTION
                    PRINT_DEBUG;
                    cerr << "\tb_r_k out of bounds" << endl;
                    exit(1);
            }
            uint32_t pos = POS_MASK[bit_pos];
            uint32_t board_at_pos = board & pos;
            uint32_t forwd4 = board & FORWD_4(BLK,pos);
            uint32_t forwd3_5 = board & FORWD(BLK,pos) & ~forwd4;
            uint32_t bckwd4 = board & BCKWD_4(BLK,pos);
            uint32_t bckwd3_5 = board & BCKWD(BLK,pos) & ~bckwd4;
            uint32_t jump_forwd_r = 0;
            uint32_t jump_forwd_l = 0;
            uint32_t jump_bckwd_r = 0;
            uint32_t jump_bckwd_l = 0;
            if (c < 3){
                jump_forwd_r = board & COL_MASK(c+1) & FORWD_JUMP(BLK, pos);
                jump_bckwd_r = board & COL_MASK(c+1) & BCKWD_JUMP(BLK, pos);
            }
            if (c > 0){
                jump_forwd_l = board & COL_MASK(c-1) & FORWD_JUMP(BLK, pos);
                jump_bckwd_l = board & COL_MASK(c-1) & BCKWD_JUMP(BLK, pos);
            }

            //cout << "board " << b_r_k << endl;
            //cout << "pos " << bit_pos << endl;
            //print_board(board_at_pos);
            neuron_input.push_back((bool) board_at_pos);
            //cout << "forwd" << endl;
            //print_board(forwd4);
            neuron_input.push_back((bool) forwd4);
            //print_board(forwd3_5);
            neuron_input.push_back((bool) forwd3_5);
            //cout << "bckwd" << endl;
            //print_board(bckwd4);
            neuron_input.push_back((bool) bckwd4);
            //print_board(bckwd3_5);
            neuron_input.push_back((bool) bckwd3_5);
            //cout << "jump_forwd" << endl;
            //print_board(jump_forwd_r);
            neuron_input.push_back((bool) jump_forwd_r);
            //print_board(jump_forwd_l);
            neuron_input.push_back((bool) jump_forwd_l);
            //cout << "jump_bckwd" << endl;
            //print_board(jump_bckwd_r);
            neuron_input.push_back((bool) jump_bckwd_r);
            //print_board(jump_bckwd_l);
            neuron_input.push_back((bool) jump_bckwd_l);
            //cout << "------------------" << endl;
        }
        all_neurons.push_back(neuron_input);
    }
}

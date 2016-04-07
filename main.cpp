// Release 0.2
// Adam Levy

#include <iostream>
#include <sys/stat.h>
#include <fstream>
#include <random>
#include <chrono>

using std::cout;
using std::endl;

#include "checkerboard.hpp"
#include "minimax_static.hpp"

typedef std::default_random_engine def_rand_eng;
typedef std::uniform_int_distribution<int> unif_dist;
typedef std::bernoulli_distribution bern_dist;

int write_board(std::ofstream & of, const BitBoard & bb)
{
    of.write((char *)(&bb), 3 * sizeof(uint32_t));
    if (of.good()){
        return 0;
    }
    return -1;
}

#pragma pack(push, 1)
struct GameStat
{
    bool win = false;
    uint8_t num_moves = 0;
};
#pragma pack(pop)

int write_game(std::ofstream & of, const bool & win, const size_t & num_moves)
{
    GameStat gstat;
    gstat.win = win;
    gstat.num_moves = num_moves;
    of.write((char *)(&gstat), sizeof(GameStat));
    if (of.good()){
        return 0;
    }
    return -1;
}

bool fileExists(const char* file) {
    struct stat buf;
    return (stat(file, &buf) == 0);
}

void gen_100gamedata()
{
    std::ofstream binfile;
    std::ofstream infofile;
    static size_t file_num = 0;
    char str1[40];
    char str2[40];
    sprintf(str1, "./game_data/games%04lu.bin", file_num);
    sprintf(str2, "./game_data/games%04lu.info", file_num++);
    while (fileExists(str1) || fileExists(str2)){
        sprintf(str1, "./game_data/games%04lu.bin", file_num);
        sprintf(str2, "./game_data/games%04lu.info", file_num++);
    }

    binfile.open(str1, std::ios::out | std::ios::binary);
    infofile.open(str2, std::ios::out | std::ios::binary);

    if (!binfile.is_open() || !infofile.is_open()){
        cout << "Failed to open file" << endl;
        exit(1);
    } else{
        cout << str1 << endl;
        cout << str2 << endl;
    }

    size_t red_win = 0;
    size_t blk_win = 0;

    size_t moves = 0;
    size_t rand_moves = 0;

    size_t num_children = 0;
    size_t num_eval = 0;

    for (size_t game = 0; game < 100; game++){
        size_t game_moves = 0;
        BitBoard move;
        Minimax_static mm(move, 15);
        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        def_rand_eng gen(seed);
        bern_dist bdist(.5);

        bool has_move = true;

        while (has_move){
            bool pick_rand = bdist(gen);

            // random move
            if (pick_rand){
                cout << ".";
                auto children = gen_children(move);
                num_children += children.size();
                num_eval++;
                if (children.size()){
                    write_board(binfile, move);
                    game_moves++;
                    unif_dist udist(0,children.size() - 1);
                    int rand_index = udist(gen);
                    move = children[rand_index];
                    moves++;
                    rand_moves++;
                    //cout << moves << ": " << "rand" << endl;
                    //print_bb(move);
                } else{
                    has_move = false;
                }
            } else{     // minimax move
                mm.set_root_node(move);
                BitBoard newMove = mm.evaluate(10);
                if (newMove != move){
                    write_board(binfile, move);
                    game_moves++;
                    moves++;
                    move = newMove;
                    //cout << moves << ": " << "minimax" << endl;
                    //print_bb(move);
                } else{
                    has_move = false;
                }
            }
        }
        cout << endl;

        bool win;
        if (move.turn == BLK){
            red_win++;
            win = false;
        } else{
            blk_win++;
            win = true;
        }

        //print_bb(move);
        //cout << win << endl;

        write_game(infofile, win, game_moves);
    }

    cout << "Game Data Summary" << endl;
    cout << "file_num: " << file_num - 1 << endl;
    cout << "#  Rand: " << rand_moves << endl;
    cout << "# Moves: " << moves << endl;
    double avg_moves = (double)moves / 100;
    cout << "avg moves per game: " << avg_moves << endl;
    double avg_branch = (double)num_children / num_eval;
    cout << "avg branch factor: " << avg_branch << endl;
    double p = (double)rand_moves / moves;
    cout << "% rand moves: " << p << endl;
    double rw = (double)red_win / (red_win + blk_win);
    cout << "% Red Wins: " << rw << endl;
    cout << endl;

    binfile.close();
    infofile.close();
}

int main(){

    for(size_t i = 0; i < 10; i++){
        gen_100gamedata();
    }

    return 0;
}

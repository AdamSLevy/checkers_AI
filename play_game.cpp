#include <string>
using std::string;

#include <iostream>
using std::cout;
using std::endl;

#include <stdexcept>
#include <chrono>

// SKYNET stuff
#include "skynet/src/skynet/checkers_client.hpp"
using skynet::checkers::play_game;
using skynet::checkers::info_game;
using skynet::checkers::game_info_t;
using skynet::checkers::status_t;
using skynet::checkers::board_t;

#include "minimax_mcmc.hpp"

#include <random>
typedef std::default_random_engine def_rand_eng;
typedef std::uniform_int_distribution<int> unif_dist;
typedef std::bernoulli_distribution bern_dist;

BitBoard make_move(Minimax_mcmc & mm, const BitBoard & bb);

int main(int argc, char *argv[])
{
    bool player = RED;
    string game_name;
    string server_name("skynet.cs.uaf.edu");

    // Parse game name and player from cmd line args
    string player_string;
    if (argc > 2){
        string g_name(argv[1]);
        game_name = g_name;
        player_string = string(argv[2]);
        if (player_string == "R" || player_string == "r"){
            player = RED;
            player_string = "Red";
        } else if (player_string == "B" || player_string == "b"){
            player = BLK;
            player_string = "Black";
        } else{
            cout << "Invalid player!" << endl;
            cout << "Specify game name and player!" << endl;
            cout << "./play_game [name] [R/B] [server (optional, default: skynet.cs.uaf.edu)]" << endl;
            return 1;
        }
    } else{
        cout << "Specify game name and player!" << endl;
        cout << "./play_game [name] [R/B] [server (optional, default: skynet.cs.uaf.edu)]" << endl;
        return 1;
    }

    // optional server arg
    if (argc == 4){
        string s_name(argv[3]);
        server_name = s_name;
    }

    cout << "Server:    " << server_name << endl;
    cout << "Game Name: " << game_name << endl;

    BitBoard start_board;
    start_board.turn = RED;


    vector<BitBoard> children;
    string board_string = to_string(start_board);

    bool game_over = false;
    Minimax_mcmc mm(start_board, 15);
    try{
        string last_game_string;
        while(true){
            // Get game info
            game_info_t game_info = info_game(server_name, game_name);
            string game_string = game_info.boards.back();
            status_t game_status = game_info.status;

            bool turn;
            string turn_string;
            if (game_status == skynet::checkers::RED_TURN){
                turn = RED;
                turn_string = "Red";
            } else if (game_status == skynet::checkers::BLACK_TURN){
                turn = BLK;
                turn_string = "Black";
            } else{
                string winner;
                if (game_status == skynet::checkers::RED_WON){
                    winner = "Red";
                } else{
                    winner = "Black";
                }
                cout << "Game Over! " << winner << " won." << endl;
                exit(0);
            }

            BitBoard server_board = from_string(game_string, turn);

            // Print board
            if (game_string != last_game_string){
                cout << "Playing as " << player_string << endl;
                cout << "Turn: " << turn_string << endl;
                print_bb(server_board);
                last_game_string = game_string;
            } else{
                sleep(2);
            }

            if (player == turn){
                // Make sure opponent move is valid
                if (children.size() > 0){
                    bool valid = false;
                    for (auto bb : children){
                        if (bb == server_board){
                            valid = true;
                            break;
                        }
                    }
                    if (!valid){
                        cout << "INVALID MOVE!" << endl;
                        exit(1);
                    }
                }

                // Select move
                BitBoard move = make_move(mm,server_board);
                print_bb(move);
                string move_string = to_string(move);
                cout << move_string << endl;
                play_game(server_name, game_name, move_string);
                cout << "played.." << endl;
                children = gen_children(move);
            }

            sleep(1);
        }
    }
    catch(std::exception & error)
    {
        cout << "Error - " << error.what() << endl;
    }
    catch(...)
    {
        cout << "Error - Unknown." << endl;
    }


    return 0;
}

BitBoard make_move(Minimax_mcmc & mm, const BitBoard & bb)
{
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    def_rand_eng gen(seed);
    bern_dist bdist(.1);

    bool pick_rand = bdist(gen);

    BitBoard move;
    // random move
    auto children = gen_children(bb);
    if (children.size() == 0){
        cout << "Error, no moves available!" << endl;
        exit(1);
    }
    if (false /*pick_rand*/){
        unif_dist udist(0,children.size() - 1);
        int rand_index = udist(gen);
        move = children[rand_index];
        //cout << moves << ": " << "rand" << endl;
        //print_bb(move);
    } else{     // minimax move
        mm.set_root_node(bb);
        move = mm.evaluate(1);
    }

    return move;
}

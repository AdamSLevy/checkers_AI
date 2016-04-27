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

#include "mcmc.h"
#include <thrust/sort.h>
#define MAX_NUM_MOVES BB_PRE_ALLOC
#define MAX_NUM_REPEAT 1
#define NUM_ITERS 1024

vector<BitBoard> make_move(const BitBoard & bb, ullong * d_wins, BitBoard_gpu * d_boards, curandState * d_state, bool player);

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

    curandState * d_state;
    // Set up randstate
    checkCudaErrors(cudaMalloc((void **)&d_state, 
                        MAX_NUM_MOVES * MAX_NUM_REPEAT * NUM_ITERS * sizeof(curandState)));
    ullong time = system_clock::to_time_t(system_clock::now());
    dim3 blocks(MAX_NUM_MOVES, MAX_NUM_REPEAT);
    setup_kernel<<<blocks,NUM_ITERS>>>(d_state, time);
    checkCudaErrors(cudaDeviceSynchronize());

    // Set up space for boards to be evaluated
    BitBoard_gpu * d_boards;
    checkCudaErrors(cudaMalloc(&d_boards, MAX_NUM_MOVES * sizeof(BitBoard_gpu)));

    // Set up space and zero win count
    ullong * d_wins;
    checkCudaErrors(cudaMalloc(&d_wins, MAX_NUM_MOVES * sizeof(ullong)));
    checkCudaErrors(cudaMemset(d_wins, 0, MAX_NUM_MOVES * sizeof(ullong)));
    checkCudaErrors(cudaDeviceSynchronize());
    cout << "sizeof(curandState) " << sizeof(curandState) << endl;

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
                usleep(300e3);
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
                auto moves = make_move(server_board, d_wins, d_boards, d_state, player);
                auto move = moves.back();
                print_bb(move);
                string move_string = to_string(move);
                cout << move_string << endl;
                play_game(server_name, game_name, move_string);
                cout << "played.." << endl;
                children = gen_children(move);
                setup_kernel<<<blocks,NUM_ITERS>>>(d_state, time);
            }

            usleep(300e3);
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

vector<BitBoard> make_move(const BitBoard & bb, ullong * d_wins, BitBoard_gpu * d_boards, curandState * d_state, bool player)
{
    ullong start_time = system_clock::to_time_t(system_clock::now());
    auto children = gen_children(bb);
    int num_children = children.size();

    if (num_children <= 1){
        return children;
    }

    // Copy child boards to device
    checkCudaErrors(cudaMemcpy(d_boards,
                        &children[0],
                        num_children * sizeof(BitBoard),
                        cudaMemcpyHostToDevice));

    // Zero out win counts
    checkCudaErrors(cudaMemset(d_wins, 0, num_children * sizeof(ullong)));

    int num_ki = MAX_NUM_MOVES * MAX_NUM_REPEAT / num_children;

    cout << num_children << ", " << num_ki << endl;
    dim3 blocks(num_children,num_ki);
    ullong h_wins[num_children];
    ullong time = system_clock::to_time_t(system_clock::now());
    ullong elapsed_time;
    ullong limit = 10;
    size_t num_iter = 0;
    while (time - start_time < limit){
        random_descent<<<blocks,NUM_ITERS>>>(d_state, d_boards, d_wins, player);
        num_iter++;
        checkCudaErrors(cudaDeviceSynchronize());
        // Sort boards by win count, lowest to highest
        ullong new_time = system_clock::to_time_t(system_clock::now());
        if (new_time - start_time < 15 - elapsed_time){
            checkCudaErrors(cudaMemcpy(h_wins,
                                d_wins,
                                num_children * sizeof(ullong),
                                cudaMemcpyDeviceToHost));
        }
        elapsed_time = new_time - time;
        limit = 15 - elapsed_time;
        time = new_time;
        cout << "Time elapsed: " << time - start_time << endl;
    }
    thrust::sort_by_key(h_wins, h_wins + num_children, &children[0]);

    cout << "num playouts " << num_iter * num_ki * NUM_ITERS << endl;
    for(int i = 0; i < num_children; i++){
        cout << (double) h_wins[i] / (num_iter * num_ki * NUM_ITERS) << endl;
    }
    cout << endl;

    return children;
}

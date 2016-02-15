#include <iostream>
using std::cout;
using std::endl;

#include <stdio.h>

#include <time.h>

#include "checkerboard.hpp"
#include "ffnn.hpp"


int main(){

    printf("Timing CheckerBoard::gen_children with psuedo-randomized boards\n");
    cout << "wait ... (~30 sec)" << endl;

    CheckerBoard ckr;
    BitBoard bb;
    unsigned long long num_calls = 0;

    clock_t start_time = clock();
    while (num_calls < 1e7){
        ckr.gen_children();
        size_t num_children = ckr.m_children.size();
        if (num_children == 0){
            ckr.m_bb = bb;
        } else{
            ckr.m_bb = ckr.m_children[num_calls % num_children];
        }
        ckr.m_children.clear();
        num_calls++;
    }
    clock_t elapsed = clock() - start_time;
    printf("%llu function calls\n", num_calls);
    double seconds_elapsed = (double)elapsed/CLOCKS_PER_SEC;

    printf("%lu clock cycles elapsed\n", elapsed);
    printf("%f seconds elapsed\n", seconds_elapsed);

    printf("%f calls per sec\n", num_calls / seconds_elapsed);
    printf("%f nsec per call\n", (seconds_elapsed * 1e9) / num_calls);

    cout << endl;

    printf("Timing FFNN::forward_pass() with randomly initialized weights and pseudo_random boards.\n\
            Highest value board is followed for BLK. Lowest is followed for RED. \n\
            Child boards are evaluated simultaneously.\n");


    int x = 2;
    while (x > 0){
        vector<size_t> net_lo = {32, 40, 10, 1};
        FFNN net(net_lo);
        num_calls = 0;
        unsigned long long num_boards = 0;
        ckr.m_bb = bb;
        ckr.m_children.clear();
        mat out;
        mat input;
        if(x == 1){
            net_lo = {32, 80, 70, 50, 1};
            net = FFNN(net_lo);
            printf("Network Structure: 32 -> 80 -> 70 -> 50 -> 1\n");
            printf("Total num nodes: = %d\n", 32*80 + 80*70 + 70 * 50 + 50);
        } else{
            printf("Network Structure: 32 -> 40 -> 10 -> 1\n");
            printf("Total num nodes: = %d\n", 32*40 + 40*10 + 10);
        }
        cout << "wait ... (~30 sec)" << endl;

        start_time = clock();
        while (num_calls < 1e5){
            ckr.gen_children();
            size_t num_children = ckr.m_children.size();
            if (num_children == 0){
                ckr.m_bb = bb;
            } else{
                // create input matrix
                input = gen_input_mat(ckr.m_children);

                // compute output
                out = net.forward_pass(input);
                //net.set_input(input);
                //net.forward_pass();
                //out = net.output();

                //cout << "input: \n";
                //cout << input << endl;
                //cout << "out: \n";
                //cout << out << endl;

                // find max
                arma::uword move_id;
                if(ckr.m_bb.turn == BLK){
                    out.max(move_id);
                    //cout << "max id: " << move_id << endl;
                } else{
                    out.min(move_id);
                    //cout << "min id: " << move_id << endl;
                }

                // select board
                ckr.m_bb = ckr.m_children[move_id];
            }
            ckr.m_children.clear();
            num_calls++;
            num_boards += num_children;
        }
        elapsed = clock() - start_time;
        elapsed = clock() - start_time;
        seconds_elapsed = (double)elapsed/CLOCKS_PER_SEC;

        printf("%llu function calls\n", num_calls);
        printf("%llu boards evaluated\n", num_boards);

        printf("%lu clock cycles elapsed\n", elapsed);
        printf("%f seconds elapsed\n", seconds_elapsed);

        printf("%f calls per sec\n", num_calls / seconds_elapsed);
        printf("%f boards per sec\n", num_boards / seconds_elapsed);
        printf("%f nsec per call\n", (seconds_elapsed * 1e9) / num_calls);
        printf("%f nsec per board\n", (seconds_elapsed * 1e9) / num_boards);

        cout << endl;
        x--;
    }


    return 0;
}

# Checkers
This project is a checkers representation and AI by Adam S. Levy, written for Professor Genetti's Spring 2016 CS 405: "Intro to AI" class at University of Alaska, Fairbanks.
The CheckerBoard class has member functions for generating possible moves from a given board state.
The BitBoard struct is a memory efficient board representation using 3 uint32_t bit boards. 

## Branch: feature/ffnn
This branch features a fully connected Feed Forward Neural Network (FFNN) and a separate main function for timing (timing.cpp).

## Build instructions
The FFNN class depends on the Armadillo C++ matrix library. More information can be found at [http://arma.sourceforge.net/download.html](http://arma.sourceforge.net/download.html).
For timing, run `make time`. Then run `./time`.
For a demonstration of the board generation function just run `make`. Then run `./checkers`.


### Timing Output
Below is the output from the timing program. See timing.cpp for the code used to generate this data and output.

Note that for FFNN::forward_pass(), nsec per call includes the time to
- generate child boards, 
- generate the input matrix for all child boards, 
- compute the outputs, 
- find the max, 
- and select the appropriate child board.

Also note that nsec per board is much lower than nsec per call since multiple child boards are evaluated in a single forward_pass() call.
Each of the child boards becomes a row of the input matrix. This way these boards are evaluated simultaneously in a single matrix operation.

For comparison, on [skynet](http://skynet.cs.uaf.edu) the "Blondie 24" network (32, 40, 10, 1) with all ones for inputs and weights,
computes forward pass every 19844 nsec or 50393 board evaluations per second. This of course is only doing one board per evaluation.

#### Output from timing.cpp:

```
Timing CheckerBoard::gen_children with psuedo-randomized boards
wait ... (~30 sec)
10000000 function calls
1245448 clock cycles elapsed
1.245448 seconds elapsed
8029239.277754 calls per sec
124.544800 nsec per call

Timing FFNN::forward_pass() with randomly initialized weights and pseudo_random boards.
            Highest value board is followed for BLK. Lowest is followed for RED. 
            Child boards are evaluated simultaneously.
Network Structure: 32 -> 40 -> 10 -> 1
Total num nodes: = 1690
wait ... (~30 sec)
100000 function calls
674941 boards evaluated
1175763 clock cycles elapsed
1.175763 seconds elapsed
85051.154017 calls per sec
574045.109431 boards per sec
11757.630000 nsec per call
1742.023377 nsec per board

Network Structure: 32 -> 80 -> 70 -> 50 -> 1
Total num nodes: = 11710
wait ... (~30 sec)
100000 function calls
466054 boards evaluated
3574629 clock cycles elapsed
3.574629 seconds elapsed
27974.931105 calls per sec
130378.285411 boards per sec
35746.290000 nsec per call
7669.988885 nsec per board

```

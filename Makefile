CXX=g++
CXXFLAGS=-Wall -std=c++11 -O3

checkers: main.o checkerboard.o minimax_static.o static_eval.o
	g++ -O3 main.o checkerboard.o minimax_static.o static_eval.o -o checkers
main.o: main.cpp checkerboard.hpp minimax_static.hpp
checkerboard.o: checkerboard.cpp checkerboard.hpp bit_mask_init.h
minimax_static.o: minimax_static.cpp minimax_static.hpp static_eval.hpp
static_eval.o: static_eval.cpp static_eval.hpp

clean:
	rm *.o

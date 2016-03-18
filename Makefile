CXX=g++
CXXFLAGS=-Wall -std=c++11 -O3

checkers: main.o checkerboard.o ffnn.o minimax.o
	g++ -O3 main.o checkerboard.o ffnn.o minimax.o -o checkers -larmadillo
time: timing.o checkerboard.o ffnn.o
	g++ -O3 timing.o checkerboard.o ffnn.o -o time -larmadillo
main.o: 		main.cpp 			checkerboard.hpp 	ffnn.hpp minimax.hpp 	bit_mask_init.h
timing.o: 		timing.cpp 			checkerboard.hpp 	bit_mask_init.h
minimax.o: 		minimax.cpp 		minimax.hpp 		ffnn.hpp 				checkerboard.hpp
checkerboard.o: checkerboard.cpp 	checkerboard.hpp 	bit_mask_init.h
ffnn.o: 		ffnn.cpp 			ffnn.hpp

clean:
	rm *.o

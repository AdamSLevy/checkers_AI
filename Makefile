CXX=g++
CXXFLAGS=-Wall -std=c++11 -O2

checkers: main.o checkerboard.o
	g++ main.o checkerboard.o ffnn.o -o checkers -larmadillo
main.o: checkerboard.o ffnn.o
checkerboard.o: checkerboard.cpp checkerboard.hpp bit_mask_init.h
ffnn.o: ffnn.cpp ffnn.hpp

clean:
	rm *.o

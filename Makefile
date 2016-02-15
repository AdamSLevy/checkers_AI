CXX=g++
CXXFLAGS=-Wall -std=c++11 -O3

checkers: main.o checkerboard.o ffnn.o
	g++ -O3 main.o checkerboard.o ffnn.o -o checkers -larmadillo
time: timing.o checkerboard.o ffnn.o
	g++ -O3 timing.o checkerboard.o ffnn.o -o time -larmadillo
main.o: checkerboard.hpp bit_mask_init.h ffnn.hpp
timing.o: timing.cpp checkerboard.hpp bit_mask_init.h
checkerboard.o: checkerboard.cpp checkerboard.hpp bit_mask_init.h
ffnn.o: ffnn.cpp ffnn.hpp

clean:
	rm *.o

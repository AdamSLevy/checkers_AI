CXX=g++
CXXFLAGS=-Wall -std=c++11

checkers: main.o checkerboard.o
	g++ main.o checkerboard.o -o checkers
main.o: checkerboard.o
checkerboard.o: checkerboard.cpp checkerboard.h bit_mask_init.h

clean:
	rm *.o

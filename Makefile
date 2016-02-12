CXX=g++
CXXFLAGS=-Wall -std=c++11

checkers: main.o checkerboard.o convnet.o
	g++ main.o checkerboard.o convnet.o -o checkers
main.o: checkerboard.o
convnet.o: convnet.cpp convnet.h checkerboard.h bit_mask_init.h
checkerboard.o: checkerboard.cpp checkerboard.h bit_mask_init.h

clean:
	rm *.o

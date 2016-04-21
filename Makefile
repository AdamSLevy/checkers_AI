CXX=g++
CXXFLAGS=-Wall -std=c++11

cudnn: main.cu
	nvcc -std=c++11 main.cu -L/usr/local/cuda/lib -lcudnn -lcublas
checkers: main.o checkerboard.o
	g++ main.o checkerboard.o -o checkers
main.o: checkerboard.o
checkerboard.o: checkerboard.cpp checkerboard.hpp bit_mask_init.h

clean:
	rm *.o

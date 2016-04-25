CXX=g++
CXXFLAGS=-Wall -std=c++11

mcmc: mcmc.o checkerboard.o checkerboard_gpu.o
	nvcc -std=c++11 mcmc.o checkerboard.o checkerboard_gpu.o -L/usr/local/cuda/lib -lcurand
mcmc.o: mcmc.cu
	nvcc -std=c++11 -c mcmc.cu -rdc=true
checkerboard_gpu.o: checkerboard_gpu.cu checkerboard_gpu.hpp
	nvcc -std=c++11 -c checkerboard_gpu.cu -rdc=true
cudnn: main.cu
	nvcc -std=c++11 main.cu -L/usr/local/cuda/lib -lcudnn -lcublas
checkers: main.o checkerboard.o
	g++ main.o checkerboard.o -o checkers
main.o: checkerboard.o
checkerboard.o: checkerboard.cpp checkerboard.hpp bit_mask_init.h

clean:
	rm *.o

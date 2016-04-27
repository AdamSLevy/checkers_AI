CXX=g++

SRC=skynet/src
INC=skynet/include
CXXFLAGS= -O3 -std=c++11
CFLAGS=-O3 -std=c++11 -I$(INC) -I$(SRC)
LIB=

JSON_SRC=$(INC)/jsoncpp/json_reader.cpp $(INC)/jsoncpp/json_value.cpp $(INC)/jsoncpp/json_writer.cpp $(SRC)/json/json.cpp
MONGOOSE_SRC=$(INC)/mongoose/mongoose.c
SKYNET_SRC=$(SRC)/skynet/checkers.cpp $(SRC)/skynet/checkers_client.cpp $(SRC)/skynet/neuralnet.cpp
play_game: play_game.o mcmc.o checkerboard.o checkerboard_gpu.o $(JSON_SRC) $(MONGOOSE_SRC) $(SKYNET_SRC)
	nvcc $(CFLAGS) $^ -L/usr/local/cuda/lib -lcurand -o $@
mcmc.o: mcmc.cu mcmc.h
	nvcc -O3 -std=c++11 -c mcmc.cu -rdc=true
checkerboard_gpu.o: checkerboard_gpu.cu checkerboard_gpu.hpp
	nvcc -O3 -std=c++11 -c checkerboard_gpu.cu -rdc=true
checkerboard.o: checkerboard.cpp checkerboard.hpp bit_mask_init.h
play_game.o: play_game.cu
	nvcc -O3 -std=c++11 -c play_game.cu

#checkers: main.o checkerboard.o
#	g++ main.o checkerboard.o -o checkers
#main.o: checkerboard.o
#cudnn: main.cu
#	nvcc -std=c++11 main.cu -L/usr/local/cuda/lib -lcudnn -lcublas

clean:
	rm *.o a.out

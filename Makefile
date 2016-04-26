CXX=g++

SRC=skynet/src
INC=skynet/include
CXXFLAGS=-std=c++11
CFLAGS=-O -std=c++11 -I$(INC) -I$(SRC)
LIB=

JSON_SRC=$(INC)/jsoncpp/json_reader.cpp $(INC)/jsoncpp/json_value.cpp $(INC)/jsoncpp/json_writer.cpp $(SRC)/json/json.cpp
MONGOOSE_SRC=$(INC)/mongoose/mongoose.c
SKYNET_SRC=$(SRC)/skynet/checkers.cpp $(SRC)/skynet/checkers_client.cpp $(SRC)/skynet/neuralnet.cpp
play_game: play_game.o minimax_mcmc.o mcmc.o checkerboard.o checkerboard_gpu.o $(JSON_SRC) $(MONGOOSE_SRC) $(SKYNET_SRC)
	nvcc $(CFLAGS) $^ -L/usr/local/cuda/lib -lcurand -o $@
minimax_mcmc.o: minimax_mcmc.cu minimax_mcmc.hpp mcmc.h
	nvcc -std=c++11 -c minimax_mcmc.cu
mcmc.o: mcmc.cu mcmc.h
	nvcc -std=c++11 -c mcmc.cu -rdc=true
checkerboard_gpu.o: checkerboard_gpu.cu checkerboard_gpu.hpp
	nvcc -std=c++11 -c checkerboard_gpu.cu -rdc=true
checkerboard.o: checkerboard.cpp checkerboard.hpp bit_mask_init.h
play_game.o: play_game.cpp
	nvcc -std=c++11 -c play_game.cpp

#checkers: main.o checkerboard.o
#	g++ main.o checkerboard.o -o checkers
#main.o: checkerboard.o
#cudnn: main.cu
#	nvcc -std=c++11 main.cu -L/usr/local/cuda/lib -lcudnn -lcublas

clean:
	rm *.o a.out

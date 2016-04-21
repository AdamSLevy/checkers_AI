SRC=skynet/src
INC=skynet/include
CXX=g++
CXXFLAGS=-std=c++11
CFLAGS=-O -std=c++11 -Wall -Wno-unused-result -Wno-deprecated-register -I$(INC) -I$(SRC)
LIB=

JSON_SRC=$(INC)/jsoncpp/json_reader.cpp $(INC)/jsoncpp/json_value.cpp $(INC)/jsoncpp/json_writer.cpp $(SRC)/json/json.cpp
MONGOOSE_SRC=$(INC)/mongoose/mongoose.c
SKYNET_SRC=$(SRC)/skynet/checkers.cpp $(SRC)/skynet/checkers_client.cpp $(SRC)/skynet/neuralnet.cpp

#CXXFLAGS=-Wall -std=c++11

play_game: play_game_online.o checkerboard.o $(JSON_SRC) $(MONGOOSE_SRC) $(SKYNET_SRC)
	$(CXX) $(CFLAGS) $(LIB) $^ -o $@
checkers: main.o checkerboard.o
	g++ main.o checkerboard.o -o checkers
main.o: checkerboard.o
play_games_online.o: checkerboard.o checkers_client.o
checkerboard.o: checkerboard.cpp checkerboard.hpp bit_mask_init.h
checkers_client.o: $(SRC)/skynet/checkers_client.cpp $(SRC)/skynet/checkers_client.hpp $(JSON_SRC) $(MONGOOSE_SRC) #$(SKYNET_SRC)

clean:
	rm *.o



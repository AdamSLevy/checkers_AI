# Checkers :: Monte Carlo Random Descent Playouts
#### Adam Levy // CS405 Intro to AI // Professor Jon Genetti // UAF Spring 2016
A Checkers program that plays on the `http:://skynet.uaf.cs.edu/` game server.

## Dependencies
- Cuda capable GPU
- Cuda Developer Toolkit 7.0 or greater
- C++11
- UAF's Skynet code

## Build Set Up
You need to find where your cuda library lives so you can link it.
Edit line 13 in the Makefile. My path is `/usr/local/cuda/lib`
```
play_game: play_game.o mcmc.o checkerboard.o checkerboard_gpu.o $(JSON_SRC) $(MONGOOSE_SRC) $(SKYNET_SRC)
    nvcc $(CFLAGS) $^ -L/usr/local/cuda/lib -lcurand -o $@
```
If you cloned from git you will also have to download the submodule. Run
```
$ git submodule update
```

## Build
Just run `make`.

## Run
The program attempts to connect to the `http:://skynet.uaf.cs.edu/` game
server. You must pass the name of the game on the server and the color
you are playing as using `r` or `b`. Red goes first on the game server.
The program will exit when the game is over.
```
$ ./play_game game1 r
```

#!/bin/bash                                                                                                                                                                                                   
COMPILO = /usr/bin/g++ -O3 -ffast-math -mfma -fno-trapping-math -march=native -mtune=native -mavx -std=c++11 -mprefer-vector-width=512 -fopenmp -lpthread # -fopenacc # -fopenmp-simd 
DEBUG   = #-g -fopt-info-optall -fverbose-asm -Wall                                                                                                                                                          
INCLUDE = -lm -lpthread -lblas -llapack -lfftw3 

interp_light : interp_light.o makefile
	$(COMPILO) $(DEBUG) interp_light.o -o interp_light $(INCLUDE)

interp_light.o: interp_light.cpp  ../include/*.hpp
	$(COMPILO) $(DEBUG) $(INCLUDE) -c interp_light.cpp -o interp_light.o

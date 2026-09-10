#!/bin/bash                                                                                                                                                                                                   
COMPILO = /usr/bin/g++ -O3 -ffast-math -mfma -fno-trapping-math -march=native -mtune=native -mavx -std=c++11 -mprefer-vector-width=512 -fopenmp -lpthread -std=c++17 # -fopenacc # -fopenmp-simd 
DEBUG   = #-g -fopt-info-optall -fverbose-asm -Wall                                                                                                                                                          
INCLUDE = -lm -lpthread -lblas -llapack -lfftw3 

interp_light : interp_light.o makefile
	$(COMPILO) $(DEBUG) interp_light.o -o interp_light $(INCLUDE)

interp_light.o: interp_light.cpp  ../include/*.hpp
	$(COMPILO) $(DEBUG) $(INCLUDE) -c interp_light.cpp -o interp_light.o

inverse : inverse.o makefile
	$(COMPILO) $(DEBUG) inverse.o -o inverse $(INCLUDE)

inverse.o: inverse.cpp  ../include/*.hpp
	$(COMPILO) $(DEBUG) $(INCLUDE) -c inverse.cpp -o inverse.o

shape : shape.o makefile
	$(COMPILO) $(DEBUG) shape.o -o shape $(INCLUDE)

shape.o: shape.cpp  ../include/*.hpp
	$(COMPILO) $(DEBUG) $(INCLUDE) -c shape.cpp -o shape.o

sketching : sketching.o makefile
	$(COMPILO) $(DEBUG) sketching.o -o sketching $(INCLUDE)

sketching.o: sketching.cpp  ../include/*.hpp
	$(COMPILO) $(DEBUG) $(INCLUDE) -c sketching.cpp -o sketching.o

reinterpolation_test : reinterpolation_test.o makefile
	$(COMPILO) $(DEBUG) reinterpolation_test.o -o reinterpolation_test $(INCLUDE)

reinterpolation_test.o: reinterpolation_test.cpp  ../include/*.hpp
	$(COMPILO) $(DEBUG) $(INCLUDE) -c reinterpolation_test.cpp -o reinterpolation_test.o

test_svd : test_svd.o makefile
	$(COMPILO) $(DEBUG) test_svd.o -o test_svd $(INCLUDE)

test_svd.o: test_svd.cpp  ../include/*.hpp
	$(COMPILO) $(DEBUG) $(INCLUDE) -c test_svd.cpp -o test_svd.o

test_gauss_hermite : test_gauss_hermite.o makefile
	$(COMPILO) $(DEBUG) test_gauss_hermite.o -o test_gauss_hermite $(INCLUDE)

test_gauss_hermite.o: test_gauss_hermite.cpp  ../include/*.hpp
	$(COMPILO) $(DEBUG) $(INCLUDE) -c test_gauss_hermite.cpp -o test_gauss_hermite.o

test_gauss_laguerre : test_gauss_laguerre.o makefile
	$(COMPILO) $(DEBUG) test_gauss_laguerre.o -o test_gauss_laguerre $(INCLUDE)

test_gauss_laguerre.o: test_gauss_laguerre.cpp  ../include/*.hpp
	$(COMPILO) $(DEBUG) $(INCLUDE) -c test_gauss_laguerre.cpp -o test_gauss_laguerre.o

test_gauss_legendre : test_gauss_legendre.o makefile
	$(COMPILO) $(DEBUG) test_gauss_legendre.o -o test_gauss_legendre $(INCLUDE)

test_gauss_legendre.o: test_gauss_legendre.cpp  ../include/*.hpp
	$(COMPILO) $(DEBUG) $(INCLUDE) -c test_gauss_legendre.cpp -o test_gauss_legendre.o

test_cauchy : test_cauchy.o makefile
	$(COMPILO) $(DEBUG) test_cauchy.o -o test_cauchy $(INCLUDE)

test_cauchy.o: test_cauchy.cpp  ../include/*.hpp
	$(COMPILO) $(DEBUG) $(INCLUDE) -c test_cauchy.cpp -o test_cauchy.o

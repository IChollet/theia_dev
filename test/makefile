CXX      = g++
CXXFLAGS = -O3 -march=native -mtune=native -std=c++17 -fopenmp
DEBUG    = # -g -Wall -Wextra
LDLIBS   = -llapack -lblas -lm -lpthread
DEPFLAGS = -MMD -MP

# Sanitized build: make asan (or: make SAN=1 <target>)
ifdef SAN
CXXFLAGS = -O1 -g -std=c++17 -fsanitize=address,undefined -fno-omit-frame-pointer
endif

SRCS = $(wildcard */test_*.cpp)
BINS = $(SRCS:.cpp=)

.PHONY: all asan clean
all: $(BINS)

asan:
	$(MAKE) clean
	$(MAKE) SAN=1 all

# Any test binary is built from its .cpp; header dependencies are tracked in .d files
%: %.cpp
	$(CXX) $(CXXFLAGS) $(DEBUG) $(DEPFLAGS) $< -o $@ $(LDLIBS)

clean:
	rm -f $(BINS) $(BINS:=.d)

-include $(BINS:=.d)

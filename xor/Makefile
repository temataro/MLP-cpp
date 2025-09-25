CXXFLAGS=-Wall -Wextra
CPPSTD=c++23
LIBS=-lm
OPTS=-march=native -ffast-math -O2

.PHONY: all clean

all: main

%: %.cpp
	clang++ $(CXXFLAGS) -std=$(CPPSTD) $< -o $@ $(LIBS) $(OPTS)
	./$@

sharp:
	clang-format -i *.cpp

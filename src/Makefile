CXX  = gcc
PCXX = mpic++
CXXFLAGS = -std=c++17 -O3 -Wall
LAFLAGS  = -lblas -llapack
OMPFLAGS = -fopenmp
#LAFLAGS = -I/opt/OpenBLAS/include -L/opt/OpenBLAS/lib -lopenblas

all: a_vector.out a_shift.out a_normal.out

a_vector.out: psminres_vector.cpp psminres_utils.cpp
	$(PCXX) $^ -o $@ $(CXXFLAGS) $(LAFLAGS) $(OMPFLAGS)
a_shift.out: psminres_shift.cpp psminres_utils.cpp
	$(PCXX) $^ -o $@ $(CXXFLAGS) $(LAFLAGS) $(OMPFLAGS)
a_normal.out: psminres_normal.cpp psminres_utils.cpp
	$(PCXX) $^ -o $@ $(CXXFLAGS) $(LAFLAGS) $(OMPFLAGS)

clean:
	rm -rf *.out

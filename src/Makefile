PCXX = mpic++
CXXFLAGS = -std=c++17 -O3 -Wall -fopenmp
LAFLAGS = -lblas -llapack
#LAFLAGS = -I/opt/OpenBLAS/include -L/opt/OpenBLAS/lib -lopenblas

all: a_vector.out a_shift.out

a_vector.out: psminres_vector.cpp psminres_utils.cpp
	$(PCXX) $^ -o $@ $(CXXFLAGS) $(LAFLAGS)
a_shift.out: psminres_shift.cpp psminres_utils.cpp
	$(PCXX) $^ -o $@ $(CXXFLAGS) $(LAFLAGS)

a_coo.out:       test_coo.cpp psminres_utils.cpp
	$(PCXX) $^ -o $@ $(CXXFLAGS) $(LAFLAGS)
a_csr_mpi.out:   test_csr_mpi.cpp psminres_utils.cpp
	$(PCXX) $^ -o $@ $(CXXFLAGS) $(LAFLAGS)
a_fperror.out:   test_fperror.cpp psminres_utils.cpp
	$(PCXX) $^ -o $@ $(CXXFLAGS) $(LAFLAGS)
a_mpi_raii.out:  test_mpi_raii.cpp psminres_utils.cpp
	$(PCXX) $^ -o $@ $(CXXFLAGS) $(LAFLAGS)
a_timer.out:     test_timer.cpp psminres_utils.cpp
	$(PCXX) $^ -o $@ $(CXXFLAGS) $(LAFLAGS)
a_zdotc_mpi.out: test_zdotc_mpi.cpp psminres_utils.cpp
	$(PCXX) $^ -o $@ $(CXXFLAGS) $(LAFLAGS)

clean:
	rm -rf *.out

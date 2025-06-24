#/bin/bash


mpirun -np 4 -machinefile machinefile.txt ./a_vector.out ../data/ELSES_MATRIX_VCNT40000std_A.mtx 1
echo "--------------------"
mpirun -np 4 -machinefile machinefile.txt ./a_shift.out ../data/ELSES_MATRIX_VCNT40000std_A.mtx 1
echo "--------------------"
./a_normal.out ../data/ELSES_MATRIX_VCNT40000std_A.mtx 1

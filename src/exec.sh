#/bin/bash


mpirun -np 4 -machinefile machinefile.txt ./a_vector.out ../data/ELSES_MATRIX_VCNT4000std_A.mtx 0
mpirun -np 4 -machinefile machinefile.txt ./a_vector.out ../data/ELSES_MATRIX_VCNT4000std_A.mtx 1
mpirun -np 4 -machinefile machinefile.txt ./a_vector.out ../data/ELSES_MATRIX_VCNT4000std_A.mtx 2

mpirun -np 4 -machinefile machinefile.txt ./a_shift.out ../data/ELSES_MATRIX_VCNT4000std_A.mtx 0
mpirun -np 4 -machinefile machinefile.txt ./a_shift.out ../data/ELSES_MATRIX_VCNT4000std_A.mtx 1
mpirun -np 4 -machinefile machinefile.txt ./a_shift.out ../data/ELSES_MATRIX_VCNT4000std_A.mtx 2

./a_normal.out ../data/ELSES_MATRIX_VCNT4000std_A.mtx 0
./a_normal.out ../data/ELSES_MATRIX_VCNT4000std_A.mtx 1
./a_normal.out ../data/ELSES_MATRIX_VCNT4000std_A.mtx 2

#mpirun -np 4 -machinefile machinefile.txt --output-filename ret0-vector ./a_vector.out ../data/ELSES_MATRIX_VCNT400std_A.mtx 0
#mpirun -np 4 -machinefile machinefile.txt --output-filename ret1-vector ./a_vector.out ../data/ELSES_MATRIX_VCNT400std_A.mtx 1
#mpirun -np 4 -machinefile machinefile.txt --output-filename ret2-vector ./a_vector.out ../data/ELSES_MATRIX_VCNT400std_A.mtx 2

#mpirun -np 4 -machinefile machinefile.txt --output-filename ret0-shift ./a_shift.out ../data/ELSES_MATRIX_VCNT400std_A.mtx 0
#mpirun -np 4 -machinefile machinefile.txt --output-filename ret1-shift ./a_shift.out ../data/ELSES_MATRIX_VCNT400std_A.mtx 1
#mpirun -np 4 -machinefile machinefile.txt --output-filename ret2-shift ./a_shift.out ../data/ELSES_MATRIX_VCNT400std_A.mtx 2

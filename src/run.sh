#!/bin/bash

# VCNT4000std
./exec.sh ../data/ELSES_MATRIX_VCNT4000std_A.mtx 0 > ret-VCNT4000std-0.txt
./exec.sh ../data/ELSES_MATRIX_VCNT4000std_A.mtx 1 > ret-VCNT4000std-1.txt
./exec.sh ../data/ELSES_MATRIX_VCNT4000std_A.mtx 2 > ret-VCNT4000std-2.txt

# VCNT40000std
./exec.sh ../data/ELSES_MATRIX_VCNT40000std_A.mtx 0 > ret-VCNT40000std-0.txt
./exec.sh ../data/ELSES_MATRIX_VCNT40000std_A.mtx 1 > ret-VCNT40000std-1.txt
./exec.sh ../data/ELSES_MATRIX_VCNT40000std_A.mtx 2 > ret-VCNT40000std-2.txt


# TODO
# ./a.out $1 $2 > ret$2_$1.txt
# のような書き方もできる

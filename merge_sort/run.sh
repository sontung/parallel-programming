mpiicpc -qopenmp -o out main.cc merge_sort.cc
mpirun -n 4 ./out

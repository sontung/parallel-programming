#include <cstdio>
#include <random>
#include <stdlib.h> 
#include <stdio.h> 
#include <omp.h>
#include "merge_sort.h"
#include <mpi.h>
#include <assert.h>

int main(int argc, char** argv)
{
    int ret = MPI_Init(&argc, &argv);
    if (ret != MPI_SUCCESS) {
        printf("error: could not initialize MPI\n");
        MPI_Abort(MPI_COMM_WORLD, ret);
    }
     
    int world_size, rank;
    MPI_Status stat;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int N = 100000;
    std::random_device rd; // obtain a random number from hardware
    std::mt19937 eng(rd()); // seed the generator
    std::uniform_int_distribution<> distr(0, N); // define the range
    int *data = (int *) malloc(sizeof(int)*N);
    int *data_ref = (int *) malloc(sizeof(int)*N);
    for(int n=0; n<N; n++) data[n] = distr(eng);
    if(rank == 0) {
        #pragma omp parallel for 
        for(int i = 0; i < N; i++)
              data_ref[i] = data[i];
    }
    const double t0 = omp_get_wtime();
    mergeSort(data, 0, N-1);
    const double t1 = omp_get_wtime();
    
    MPI_Bcast(data_ref, N, MPI_INT, 0, MPI_COMM_WORLD);
    const double t2 = omp_get_wtime();
    int* arr_final = mergeSortMPI(data_ref, 0, N, rank, world_size);
    const double t3 = omp_get_wtime();
    if (rank == 0) { 
        printf("Serial version took %f\n", t1-t0);
        int error = 0;
        for(int i=0; i<N; i++) error += data[i]-arr_final[i];
        printf("MPI version took %f with error = %d\n", t3-t2, error);
    }
    MPI_Finalize();
}

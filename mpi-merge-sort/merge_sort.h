#ifndef MERGE_SORT_H
#define MERGE_SORT_H

#include <cstdio>
#include <mpi.h>

void merge(int* arr, int l, int m, int r);
void mergeSort(int* arr, int l, int r);
int* mergeSortMPI(int* arr, int l, int r, const int rank, const int world_size);
#endif // MERGE_SORT_H

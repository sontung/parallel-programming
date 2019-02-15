#include "merge_sort.h" 

// Merges two subarrays of arr[].
// First subarray is arr[l..m]
// Second subarray is arr[m+1..r]
void merge(int* arr, int l, int m, int r)
{
    int i, j, k;
    int n1 = m - l + 1;
    int n2 =  r - m;

    /* create temp arrays */
    int L[n1], R[n2];

    /* Copy data to temp arrays L[] and R[] */
    for (i = 0; i < n1; i++)
        L[i] = arr[l + i];
    for (j = 0; j < n2; j++)
        R[j] = arr[m + 1 + j];

    /* Merge the temp arrays back into arr[l..r]*/
    i = 0; // Initial index of first subarray
    j = 0; // Initial index of second subarray
    k = l; // Initial index of merged subarray
    while (i < n1 && j < n2)
    {
        if (L[i] <= R[j])
        {
            arr[k] = L[i];
            i++;
        }
        else
        {
            arr[k] = R[j];
            j++;
        }
        k++;
    }

    /* Copy the remaining elements of L[], if there
       are any */
    while (i < n1)
    {
        arr[k] = L[i];
        i++;
        k++;
    }

    /* Copy the remaining elements of R[], if there
       are any */
    while (j < n2)
    {
        arr[k] = R[j];
        j++;
        k++;
    }
}

/* l is for left index and r is right index of the
   sub-array of arr to be sorted */
void mergeSort(int* arr, int l, int r)
{
    if (l < r)
    {
        int m = l+(r-l)/2;
        mergeSort(arr, l, m);
        mergeSort(arr, m+1, r);
        merge(arr, l, m, r);
    }
}

int* mergeSortMPI(int *arr, int l, int r, const int rank, const int world_size) {
    const int numbers_per_process = (r-l)/world_size;
    const int start = numbers_per_process*rank;
    const int end = start+numbers_per_process-1;

    mergeSort(arr, start, end);
    MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, &arr[0], numbers_per_process, MPI_INT, MPI_COMM_WORLD);
    if (rank == 0) merge(arr, 0, numbers_per_process-1, 2*numbers_per_process-1);
    if (rank == 1) merge(arr, 2*numbers_per_process, 3*numbers_per_process-1, 4*numbers_per_process-1);
   
    MPI_Barrier(MPI_COMM_WORLD); 
    if (rank == 1) MPI_Send(&arr[2*numbers_per_process], 2*numbers_per_process, MPI_INT, 0, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        MPI_Recv(&arr[2*numbers_per_process], 2*numbers_per_process, MPI_INT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        merge(arr, 0, 2*numbers_per_process-1, 4*numbers_per_process-1);
    }
    MPI_Bcast(arr, r-l, MPI_INT, 0, MPI_COMM_WORLD);
    return arr; 
}


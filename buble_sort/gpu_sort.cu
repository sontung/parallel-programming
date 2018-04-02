#include <iostream>
#include <math.h>
#include <vector>
#include <string>
#include <algorithm>
#include <random>
#include <chrono>
#include <stdio.h>

#define NB_THREADS 1024
#define NB_NUMBERS 200
#define NB_EXPERIMENTS 100


void print_vector(int* array, int k) {
  for(size_t i=0; i < k; i++) {
    printf("%d ", array[i]);
  }
  printf("\n");
}

bool compare_vectors(int *v1, std::vector<int> v2) {
  for(int i=0; i < v2.size(); i++) {
    if (v1[i] != v2[i]) {
      printf("wrong at %d %d vs %d\n", i, v1[i], v2[i]);
      return false;
    }
  }
  return true;
}

std::vector<int> create_vector(int N) {
  std::vector<int> res;
  for(int i=0; i < N; i++) {
    res.push_back(i);
  }
  return res;
}


__global__
void bubble_sort(int *A, int n) {
  __shared__ int end[NB_THREADS]; // termination condition for each thread
  __shared__ int race[NB_NUMBERS]; // race condition for each element of input array
  for (int u=0; u<NB_THREADS; u++) { end[u] = 0; }
  for (int v=0; v<NB_NUMBERS; v++) { race[v] = 1; }
  int temp;
  int index = threadIdx.x;
  while (1) {
    end[index] = 1;
    for (int i=0; i<n; i++) {
      while (1) {
        if ((race[i] == 1 && race[i-1] == 1) || (i == 0 && race[i] == 1)) {
          if (A[i-1] > A[i]) {
            // Block race condition
            race[i] = 0;
            race[i-1] = 0;

            // Swap
            temp = A[i-1];
            A[i-1] = A[i];
            A[i] = temp;

            // Release race condition
            end[index] = 0;
            race[i] = 1;
            race[i-1] = 1;
          }
          break;
        }
      }
    }
    if (end[index] == 1) { break; }
  }
}

int main(void) {
  for (int e=0; e<NB_EXPERIMENTS; e++) {
    // Init random input array
    int range = NB_NUMBERS;
    std::vector<int> v = create_vector(range);
    std::vector<int> v_orig = create_vector(range);
    std::random_shuffle ( v.begin(), v.end() );
    std::random_shuffle ( v.begin(), v.end() );

    int *x;
    cudaMallocManaged(&x, range*sizeof(int));
    for (int i = 0; i < range; i++) {
      x[i] = v[i];
      printf("%d %d\n", x[i], v[i]);
    }
    print_vector(x, range);

    // Sort
    bubble_sort<<<1, NB_THREADS>>>(x, range);

    cudaDeviceSynchronize();

    // Check output
    if (!compare_vectors(x, v_orig)) {
      printf("%s\n", "Wrong algorithm");
      print_vector(x, range);
      break;
    } else { printf("%s %d\n", "True algorithm", e); }

    // Free memory
    cudaFree(x);
  }
}


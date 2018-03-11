#include <iostream>
#include <math.h>
#include <vector>
#include <string>
#include <algorithm>
#include <random>
#include <chrono>

#define NB_THREADS 2


void print_vector(int* array, int k) {
  for(int i=0; i < k; i++) {
    printf("%d ", array[k]);
  }
  printf("\n");
}

bool compare_vectors(int *v1, std::vector<int> v2) {
  for(int i=0; i < v2.size(); i++) {
    if (v1[i] != v2[i]) {
      std::cout << v1[i] << '\n';
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
  int temp;
  int index = threadIdx.x;
  int index2 = index+1;
  __shared__ int end[1];
  __shared__ int all_phases[NB_THREADS];
  printf("sorting\n");
  while (end[0] != 1) {
    printf("sorting2\n");
    if ((index == 0) || (all_phases[index] == 1)) {
      end[0] = 1;
      printf("sorting3\n");
      for (int i=0; i<n-index; i++) {
        if (A[i-1] > A[i]) {
          temp = A[i-1];
          A[i-1] = A[i];
          A[i] = temp;
          end[0] = 0;
        }
        if (i > 1 && all_phases[index2] != 1) {
          all_phases[index2] = 1;
        }
      }
      for(int i=0; i < n; i++) {
        printf("%d ", A[i]);
      }
      printf("\n");
    }
  }
}

int main(void) {
  int range = 50;
  std::vector<int> v = create_vector(range);
  std::vector<int> v_orig = create_vector(range);
  std::random_shuffle ( v.begin(), v.end() );
  std::random_shuffle ( v.begin(), v.end() );

  int *x;
  cudaMallocManaged(&x, range*sizeof(int));
  for (int i = 0; i < range; i++) {
    x[i] = v[i];
  }

  bubble_sort<<<1, NB_THREADS>>>(x, range);

  cudaDeviceSynchronize();

  if (!compare_vectors(x, v_orig)) {
    printf("%s\n", "Wrong algorithm");
  } else { printf("%s\n", "True algorithm"); }

  // Free memory
  cudaFree(x);
}

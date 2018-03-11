#include <iostream>
#include <math.h>
#include <vector>
#include <string>
#include <algorithm>
#include <random>
#include <chrono>


void print_vector(std::vector<int> a_vec, std::string head) {
  std::cout << head;
  std::cout << "[ ";
  for(int i=0; i < a_vec.size(); i++) {
    std::cout << a_vec[i] << " ";
  }
  std::cout << "]\n";
}

bool compare_vectors(std::vector<int> v1, std::vector<int> v2) {
  for(int i=0; i < v1.size(); i++) {
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

std::vector<int> bubble_sort(std::vector<int> A) {
  int n = A.size();
  bool swapped = true;
  int temp;
  while (swapped) {
    swapped = false;
    for (int i = 0; i < n; i++) {
      if (A[i-1] > A[i]) {
        temp = A[i-1];
        A[i-1] = A[i];
        A[i] = temp;
        swapped = true;
      }
    }
    n--;
  }
  return A;
}

int main(int argc, char* argv[]) {
  int nb_experiments = 100;
  int max_nb_values = 10000;

  std::random_device rd;
  std::mt19937 rng(rd());
  std::uniform_int_distribution<int> uni(1, max_nb_values); // choose how many items to sort
  using clock = std::chrono::steady_clock;
  double total_time = 0;

  for (int e=0; e < nb_experiments; e++) {
    int range = uni(rng);
    std::vector<int> v = create_vector(range);
    std::vector<int> v_orig = create_vector(range);
    std::random_shuffle ( v.begin(), v.end() );
    std::random_shuffle ( v.begin(), v.end() );

    // Sorting and timing
    clock::time_point start = clock::now();
    std::vector<int> v_sorted = bubble_sort(v);
    clock::time_point end = clock::now();
    clock::duration execution_time = end - start;
    double a_time = std::chrono::duration_cast<std::chrono::microseconds>(execution_time).count();
    total_time = total_time + a_time;

    if (!compare_vectors(v_orig, v_sorted)) {
      std::cout << "Sorting algorithm is wrong" << '\n';
    }
  }
  std::cout << "Total execution time is: "<< total_time << " ms" << '\n';
}

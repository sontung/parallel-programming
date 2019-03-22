#include "prime_test.h"

int pick_co_prime(long n) {
    int res = 2;
    while (n % res == 0) res++;
    return res;
}

//c^d mod n
int cdn(int c, long d, long n) {
   int value = 1;
   c = c % n;
   while( d > 0 )
   {
      if (d & 1) value = (value*c) % n;
      d = d >> 1;
      c = (c*c) % n;
   }
   return value;
}

int cdn2(int b, long e, long m) {
    if (m == 1) return 0;
    long c = 1;
    for (long ep=0; ep<e; ep++) c = (c*b) % m;
    return c;
}

bool p_test(long n) {
    if (n <= 3) return n > 1;
    else if (n % 2 == 0 || n % 3 == 0) return false;
    int a = 2;
    while (n % a == 0) a++;

    int test = cdn(a, n-1, n);
    if (test != 1) return false;
    else return true;
}

bool slow_d_test(long n) {
    if (n <= 3) return n > 1;
    else if (n % 2 == 0 || n % 3 == 0) return false;
    long i = 5;
    while (i*i <= n) {
        if (n % i == 0 || n % (i+2) == 0) return false;
        i += 6;
    }
    return true;
}

bool fast_test(long n) {
    if (n <= 3) return n > 1;
    else if (n % 2 == 0 || n % 3 == 0) return false;
    int a = 2;
    while (n % a == 0) a++;

    int test = cdn(a, n-1, n);
    if (test != 1) return false;
    
    else {
        long i = 5;
        while (i*i <= n) {
            if (n % i == 0 || n % (i+2) == 0) return false;
            i += 6;
        }
        return true;
    }
}


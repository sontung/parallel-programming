#include <cstdio>
#include <omp.h>
#include "prime_test.h"
#include <assert.h>

int main() {
    const long bound = 130;
    long *data = (long *) _mm_malloc(sizeof(long)*bound, 4096);
    bool *primi = (bool *) _mm_malloc(sizeof(bool)*bound, 4096);
    for (long i=0; i<bound; i++) data[i] = i;
    for (long u=0; u < bound; u++) {
        //float *d_buf1 = (float *) _mm_malloc(sizeof(float)*n, 4096);
        //printf("%d\n", u);
        primi[u] = slow_d_test(data[u]);
    }
    //printf("Verfication done\n");
    //const double t0 = omp_get_wtime();
    //for (long i=2; i<bound; i++) slow_d_test(i); 
    //const double t1 = omp_get_wtime();
    //printf("Slow d test took %f\n", t1-t0);

    //vconst double t2 = omp_get_wtime();
    //for (long i=2; i<bound; i++) fast_test(i); 
    //const double t3 = omp_get_wtime();
    //printf("Fast test took %f\n", t3-t2);
    long start = 1;
    long res = 0; 
    while (start < bound) {
    long bound2;
    long *data2 = (long *) _mm_malloc(sizeof(long)*bound, 4096);
    for (long i=start; i<bound; i++) {
        data2[i] = data2[i-1]+data[i]*primi[i];
        if (data2[i] > bound) {
            bound2 = i;
            printf("Stop summing at %ld\n", i);
            break;
        }

        printf("%ld %ld %d\n", data2[i], data[i], primi[i]);
    }
    
    const double t2 = omp_get_wtime();
    long res2;
    for (long i=bound2-1; i>0; i--) {
        if (slow_d_test(data2[i])) {
            res2 = data2[i];
            break;
        }
    }
    if (res2>res) res = res2;
    const double t3 = omp_get_wtime();
    start++;

    printf("Done in %f, res = %d\n", 0, res2);
    }

    printf("Done in %f, res = %d\n", 0, res);
}






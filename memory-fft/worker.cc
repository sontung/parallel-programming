#include <mkl.h>
#include <hbwmalloc.h>
#include <cstdio>
#include <errno.h>
#include <omp.h>

//implement scratch buffer on HBM and compute FFTs, refer instructions on Lab page
void runFFTs( const size_t fft_size, const size_t num_fft, MKL_Complex8 *data, DFTI_DESCRIPTOR_HANDLE *fftHandle) {
  const long total_size = fft_size*num_fft;
  long buff_size = fft_size;
  MKL_Complex8 *buff;
  for(long t=0; t < total_size; t+=fft_size) {
      hbw_posix_memalign((void**) &buff, 4096, sizeof(MKL_Complex8)*buff_size);  
      #pragma omp parallel for 
      for(long v = 0; v < buff_size; v += 1) {
	buff[v].real = data[t+v].real;
	buff[v].imag = data[t+v].imag;
      }

      DftiComputeForward(*fftHandle, &buff[0]);

      #pragma omp parallel for 
      for(long v = 0; v < buff_size; v += 1) {
	data[t+v].real = buff[v].real;
	data[t+v].imag = buff[v].imag;
      }
      hbw_free(buff);
  }
}

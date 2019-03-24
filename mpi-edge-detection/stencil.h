#ifndef __INCLUDED_STENCIL_H__
#define __INCLUDED_STENCIL_H__

#include "image.h"
#include <omp.h>
#include <cmath>
#include <math.h> 

#define PI 3.14159265

int** simple_kernel();

template<typename P>
void ApplyStencil(ImageClass<P> & img_in, ImageClass<P> & img_out);

void sobel_filter(ImageClass<float> &img_in, ImageClass<float> &img_out, ImageClass<float> &_theta);

void non_max_suppress(ImageClass<float> &img_grad, ImageClass<float> &img_theta);

void threshold(ImageClass<float> &img);

void tracking(ImageClass<float> &img);

bool check_blob(float* &pixels_arr, int x, int y, int width);

void sobel_filter_mpi(ImageClass<float> &img_in, ImageClass<float> &img_out, ImageClass<float> &_theta, const int myFirstRow, const int myLastRow);

void non_max_suppress_mpi(ImageClass<float> &img_grad, ImageClass<float> &img_theta, const int myFirstRow, const int myLastRow);

void threshold_mpi(ImageClass<float> &img, const int myFirstRow, const int myLastRow, float max_val);

void tracking_mpi(ImageClass<float> &img, const int myFirstRow, const int myLastRow, int rank, bool first_pass);

void Convolve_mpi(ImageClass<float> & img_in, ImageClass<float> & img_out,
                  int** kernel, int first_row, int last_row);

template<typename P>
void Convolve(ImageClass<P> & img_in, ImageClass<P> & img_out, int** kernel);
#endif

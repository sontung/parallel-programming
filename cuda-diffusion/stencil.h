#ifndef __INCLUDED_STENCIL_H__
#define __INCLUDED_STENCIL_H__

#include "image.h"
#include <omp.h>
#include <cmath>
#include <math.h> 

#define PI 3.14159265

int** simple_kernel();

template<typename P>
void ApplyStencil(ImageClass<P> & img_in, ImageClass<P> & img_out, ImageClass<P> & g_image, float time_step);

template<typename P>
void Convolve(ImageClass<P> & img_in, ImageClass<P> & img_out, int** kernel);

float g_func(float number);

float arithmetic_mean(float n1, float n2);

void calculate_g_image(ImageClass<float> &img_in, ImageClass<float> &g_image);

void copy(ImageClass<float> &src, ImageClass<float> &dst);
#endif

#include "stencil.h"

using namespace std;

template<typename P>
void ApplyStencil(ImageClass<P> & img_in, ImageClass<P> & img_out, ImageClass<P> & g_image, float time_step) {
  
  const int width  = img_in.width;
  const int height = img_in.height;

  P * in  = img_in.pixel;
  P * out = img_out.pixel;
  P * g   = g_image.pixel;

  for (int i = 1; i < height-1; i++)
    for (int j = 1; j < width-1; j++) {
        float val = in[i*width+j]*(1-time_step*(arithmetic_mean(g[i*width+j], g[(i+1)*width+j]) +
                                                arithmetic_mean(g[i*width+j], g[(i-1)*width+j]) +
                                                arithmetic_mean(g[i*width+j], g[i*width+j+1]) +
                                                arithmetic_mean(g[i*width+j], g[i*width+j-1])));
        val += in[(i+1)*width+j]*time_step*arithmetic_mean(g[i*width+j], g[(i+1)*width+j]);
        val += in[(i-1)*width+j]*time_step*arithmetic_mean(g[i*width+j], g[(i-1)*width+j]);
        val += in[i*width+j+1]*time_step*arithmetic_mean(g[i*width+j], g[i*width+j+1]);
        val += in[i*width+j-1]*time_step*arithmetic_mean(g[i*width+j], g[i*width+j-1]);

        val = (val < 0   ? 0   : val);
        val = (val > 255 ? 255 : val);

        out[i*width+j] = val;
    }
  
}

template<typename P>
void Convolve(ImageClass<P> & img_in, ImageClass<P> & img_out, int** kernel) {
  const int width  = img_in.width;
  const int height = img_in.height;

  P * in  = img_in.pixel;
  P * out = img_out.pixel;

  for (int i = 1; i < height-1; i++)
    for (int j = 1; j < width-1; j++) {
      P val = kernel[0][0]*in[(i-1)*width + j-1] +
              kernel[0][1]*in[(i-1)*width + j]   +
              kernel[0][2]*in[(i-1)*width + j+1] +
	          kernel[1][0]*in[(i  )*width + j-1] +
              kernel[1][1]*in[(i  )*width + j]   +
              kernel[1][2]*in[(i  )*width + j+1] +
	          kernel[2][0]*in[(i+1)*width + j-1] +
              kernel[2][1]*in[(i+1)*width + j]   +
              kernel[2][2]*in[(i+1)*width + j+1];

      out[i*width + j] = val;
    }
}

float arithmetic_mean(float n1, float n2) {
    return (n1+n2)/2.0;
}

float g_func(float number) {
    float lambda = 3.5;
    return exp(-pow(number, 2)/2/pow(lambda, 2));
}

void calculate_g_image(ImageClass<float> &img_in, ImageClass<float> &g_image) {
    const int width  = img_in.width;
    const int height = img_in.height;
    float* in = img_in.pixel;
    float* out = g_image.pixel;

    for (int i=1; i<height-1; i++) {
        for (int j=1; j<width-1; j++) {
            float val = pow((in[(i+1)*width+j]-in[(i-1)*width+j])/2, 2) + pow((in[i*width+j+1]-in[i*width+j-1])/2, 2);
            out[i*width+j] = g_func(val);
        }
    }
}

void copy(ImageClass<float> &src, ImageClass<float> &dst) {
    const int width  = src.width;
    const int height = src.height;
    float* in = src.pixel;
    float* out = dst.pixel;

    for (int i=0; i<height; i++) {
        for (int j=0; j<width; j++) {
            out[i*width+j] = in[i*width+j];
        }
    }
}




template void Convolve(ImageClass<float> & img_in, ImageClass<float> & img_out, int** kernel);
template void ApplyStencil<float>(ImageClass<float> & img_in, ImageClass<float> & img_out, ImageClass<float> & g_image, float time_step);

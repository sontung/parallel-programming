#include <cmath>
#include <omp.h>
#include <vector>
#include <algorithm>
#include <dirent.h>
#include <string>
#include <iostream>
#include <chrono> 
#include "image.h"
#include "stencil.h"
#define P float

void gpu_func(float* in, float* out, float* g_img, int w, int h, int n_iters);

int main(int argc, char** argv) {

    if(argc < 2) {
        printf("Usage: %s {file}\n", argv[0]);
        exit(1);
    }
    int n_trials = 1;
    int n_iters = 5;

    std::string output_name = argv[1];
    std::string delimiter = "/";

    size_t pos = 0;
    std::string token;
    while ((pos = output_name.find(delimiter)) != std::string::npos) {
        token = output_name.substr(0, pos);
        output_name.erase(0, pos + delimiter.length());
    }
    

    ImageClass<P> img_in (argv[1]);
    ImageClass<P> img_out(img_in.width, img_in.height);
    ImageClass<P> g_image(img_in.width, img_in.height);
    ImageClass<P> img_in2 (argv[1]);
    ImageClass<P> img_out2(img_in2.width, img_in2.height);
    ImageClass<P> g_image2(img_in2.width, img_in2.height);
    const int width = img_in.width;
    const int height = img_in.height;

    auto start_gpu = std::chrono::high_resolution_clock::now(); 
    gpu_func(img_in.pixel, img_out.pixel, g_image.pixel, width, height, n_iters);
    auto stop_gpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = stop_gpu-start_gpu;
    printf("GPU execution done in %f ms\n", duration.count()); 
    
    auto start_cpu = std::chrono::high_resolution_clock::now(); 
    for (int t=0; t<n_iters; t++) {
        calculate_g_image(img_in2, g_image2);
        ApplyStencil(img_in2, img_out2, g_image2, t);
        copy(img_out2, img_in2);
    }
    auto stop_cpu = std::chrono::high_resolution_clock::now(); 
    std::chrono::duration<double, std::milli> duration2 = stop_cpu-start_cpu;
    printf("CPU execution done in %f ms\n", duration2.count()); 

    float* out1 = img_out.pixel;
    float* out2 = img_out2.pixel;
    float error = 0.0;
    for (int i = 0; i < height-1; i++) {
	for (int j = 0; j < width-1; j++) {
	    if (out1[i*width+j] != 0) {
		error += abs(out1[i*width+j] - out2[i*width+j]);
	    }
	}
    }
    printf("Final image abs error = %f!\n", error/width/height);

    float* out21 = g_image.pixel;
    float* out22 = g_image2.pixel;
    float error2 = 0.0;
    for (int i = 0; i < height-1; i++) {
	for (int j = 0; j < width-1; j++) {
	    error2 += abs(out21[i*width+j] - out22[i*width+j]);
	}
    }

    printf("G image abs error = %f!\n", error2/width/height);
    
    img_out2.WriteToFile("output.png");
    img_out.WriteToFile("output2.png");




}

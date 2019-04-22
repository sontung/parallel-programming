# Isotropic non linear diffusion GPU implementation 

Run `make run`

```
g++ -c main.cc image.cc stencil.cc
nvcc -c main_cuda.cu
nvcc -lgomp -lpng -o cudaapp main.o stencil.o image.o main_cuda.o
./cudaapp test-image.png
Done executing CUDA kernel
GPU execution done in 1051.474089 ms
CPU execution done in 25040.886019 ms
Final image abs error = 0.015604!
G image abs error = 0.000095!
```

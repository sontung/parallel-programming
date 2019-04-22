
run:	
	g++ -c main.cc image.cc stencil.cc
	nvcc -c main_cuda.cu
	nvcc -lgomp -lpng -o cudaapp main.o stencil.o image.o main_cuda.o
	./cudaapp test-image.png

clean:
	rm -f *.optrpt *.o stencil cudaapp *output*.png

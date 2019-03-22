CXX=mpiicpc
CXXFLAGS=-c -qopenmp
LDFLAGS=-qopenmp -lpng

OBJECTS=main.o image.o stencil.o

stencil: $(OBJECTS)
	$(CXX) $(LDFLAGS) -o stencil $(OBJECTS)

all:	stencil

run:	all
	mpirun -n 5 ./stencil /home/sontung/Downloads/blog/media2/183442043218_0.jpg.png

queue:	all
	echo 'cd $$PBS_O_WORKDIR ; ./stencil test-image.png' | qsub -l nodes=1:flat -N edgedetection

clean:
	rm -f *.optrpt *.o stencil *output.png *~ edgedetection.*

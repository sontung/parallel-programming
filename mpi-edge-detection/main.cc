#include <cmath>
#include <omp.h>
#include "image.h"
#include "stencil.h"
#include <mpi.h>
#include <vector>
#include <algorithm>


#define P float

const int nTrials = 1;
const int skipTrials = 0; // Skip first iteration as warm-up

int main(int argc, char** argv) {

  if(argc < 2) {
    printf("Usage: %s {file}\n", argv[0]);
    exit(1);
  }
  MPI_Init(NULL, NULL);
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  ImageClass<P> img_in (argv[1]);
  ImageClass<P> img_out(img_in.width, img_in.height);
  ImageClass<P> theta(img_in.width, img_in.height);

  if (world_rank == 0) {
    printf("Running 1-node implementation...\n");
    printf("  Image size: %d x %d\n", img_in.width, img_in.height);
    
    const double t0 = omp_get_wtime();
    sobel_filter(img_in, img_out, theta);
    //non_max_suppress(img_out, theta);
    //threshold(img_out);
    //tracking(img_out);
    const double t1 = omp_get_wtime();
    printf("  Done in %f\n", t1-t0);
  }
  MPI_Barrier(MPI_COMM_WORLD);

  if (world_rank == 0) printf("Running %d-node implementation...\n", world_size);
  ImageClass<P> img_out2(img_in.width, img_in.height);
  ImageClass<P> theta2(img_in.width, img_in.height);
 
  const int width = img_in.width;
  const int height = img_in.height; 
  double t2;
  double t3;
  if (world_rank == 0) t2 = omp_get_wtime();
  sobel_filter_mpi(img_in, img_out2, theta2, world_rank, world_size);
  //non_max_suppress(img_out, theta);
  //threshold(img_out);
  //tracking(img_out);
  if (world_rank == 0) t3 = omp_get_wtime();
  MPI_Barrier(MPI_COMM_WORLD);
  
  float* pixel1 = img_out.pixel;
  float* pixel2 = img_out2.pixel;
  
  double count = double(height-2)/double(world_size);
  const int first_row = 1 + int(count*world_rank);
  const int last_row  = 1 + int(count*(world_rank+1));
  const int to_send = (last_row-first_row)*width;
  P* recv_buff = NULL;
  int* disp = NULL;
  int* rcount = NULL;
  if (world_rank == 0) {
    recv_buff = (P*)_mm_malloc(sizeof(P)*width*height, 64);
    disp = (int*)malloc(world_size*sizeof(int));
    rcount = (int*)malloc(world_size*sizeof(int)); 
    int d1;
    int d2;
    int d3;
    for (int r = 0; r < world_size; r++) {
      d1 = 1 + int(count*r);
      d2 = 1 + int(count*(r+1));
      d3 = (d2-d1)*width;
      rcount[r]= d3;
    }
    disp[0] = 0;
    for (int r = 1; r < world_size; r++) {
      disp[r] = disp[r-1] + rcount[r-1];
    }
  }
  MPI_Gatherv(&pixel2[first_row*width], to_send, MPI_FLOAT, &recv_buff[width], rcount, disp, MPI_FLOAT, 0, MPI_COMM_WORLD);
  if (world_rank == 0) {
    std::vector<int> check;
    for (int i = 1; i < height-1; i++)
      for (int j = 1; j < width-1; j++) {
        pixel2[i*width+j] = recv_buff[i*width+j];
      }
    for (int i=0; i<height-1; i++) {
        if (recv_buff[i*width+1] == 0.0) {
            bool a_bool = true;
            for (int j=0; j<width-1; j++) {
                if (recv_buff[i*width+j] != 0) {
                    a_bool = false;
                    break;
                }
            }
            if (a_bool) printf("%d %d\n", i, i*width);
        }
    }
    printf("  Done in %f\n", t3-t2);
    printf("received %d %d %d %d %d\n", rcount[0],rcount[1],rcount[2],rcount[3],rcount[4]); 
    bool bool2 = true;
    for (int j=0; j<width-1; j++) {
      if (recv_buff[638*width+j] != 0) {
        bool2 = false;
        break;
      }
    }
    if (bool2) printf("line 638 all zero from rank %d\n", world_rank);
  }

  if (world_rank == 0) {
    printf("Checking and saving results...\n");
    bool validate = true;
    for (int i = 0; i < height-1; i++)
      for (int j = 0; j < width-1; j++) {
        if (pixel1[i*width+j] != pixel2[i*width+j]) {
          validate = false;
          //printf("%f %f %d %d\n", pixel1[i*width+j], pixel2[i*width+j], i, j);
          //pixel2[i*width+j] = 255;
        }
      }
    if (validate) printf("  Results matched!\n");
    else printf("  Results unmatched!\n");
   
    img_out.WriteToFile("output.png");
    img_out2.WriteToFile("output2.png");
  }
  MPI_Finalize();
}

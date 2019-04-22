#include <cmath>
#include <omp.h>
#include "image.h"
#include "stencil.h"
#include <mpi.h>
#include <vector>
#include <algorithm>
#include <dirent.h>
#include <string>

#define P float

int main(int argc, char** argv) {

    if(argc < 2) {
        printf("Usage: %s {file}\n", argv[0]);
        exit(1);
    }
    MPI_Init(NULL, NULL);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    DIR *dir;
    struct dirent *ent;
    std::vector<std::string> files;
    if ((dir = opendir(argv[1])) != NULL) {
        while ((ent = readdir (dir)) != NULL) {
            files.push_back(ent->d_name);
        }
        closedir (dir);
    } else {
        printf("Not a directory!\n");
        return EXIT_FAILURE;
    }
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    char filename[100];
    char outputname[100];
    std::size_t check_filename;
    double non_mpi_time = 0;
    double mpi_time = 0;
    int n_trials = 1;
    for (int g = 0; g < n_trials; g++) {
        for (int f = 0; f < files.size(); f++) {
            sprintf(filename, "%s%s", argv[1], files[f].c_str());
            sprintf(outputname, "%s%s", "/home/sontung/Downloads/blog/media3/", files[f].c_str());
            std::string name(filename);
            std::size_t check_filename = name.find(".png");
            if (check_filename == std::string::npos) continue;
            if (world_rank == 0) printf("Processing %s%s\n", argv[1], files[f].c_str());

            ImageClass<P> img_in (filename);
            ImageClass<P> img_out(img_in.width, img_in.height);
            ImageClass<P> theta(img_in.width, img_in.height);

            if (world_rank == 0) {
                printf("Running 1-node implementation...\n");
                printf("  Image size: %d x %d\n", img_in.width, img_in.height);

                double t0 = omp_get_wtime();
                sobel_filter(img_in, img_out, theta);
                non_max_suppress(img_out, theta);
                threshold(img_out);
                tracking(img_out);
                double t1 = omp_get_wtime();
                printf("  Done in %f\n", t1-t0);
                non_mpi_time += t1-t0;
            }
            MPI_Barrier(MPI_COMM_WORLD);

            if (world_rank == 0) printf("Running %d-node implementation...\n", world_size);
            ImageClass<P> img_out2(img_in.width, img_in.height);
            ImageClass<P> theta2(img_in.width, img_in.height);
            float* pixel1 = img_out.pixel;
            float* pixel2 = img_out2.pixel;
            const int width = img_in.width;
            const int height = img_in.height;
            double t2;
            double t3;
            double count = double(height-2)/double(world_size);
            const int first_row = 1 + int(count*world_rank);
            const int last_row  = 1 + int(count*(world_rank+1));
            const int to_send = (last_row-first_row)*width;
            if (world_rank == 0) t2 = omp_get_wtime();
            sobel_filter_mpi(img_in, img_out2, theta2, first_row, last_row);
            non_max_suppress_mpi(img_out2, theta2, first_row, last_row);

            float* max_values = (float*)malloc(sizeof(float)*world_size);
            float max_val = 0;
            for (int i = first_row; i < last_row; i++)
                for (int j = 0; j < width-1; j++) {
                    if (pixel2[i*width+j] > max_val) max_val = pixel2[i*width+j];
                }
            MPI_Allgather(&max_val, 1, MPI_FLOAT, max_values, 1, MPI_FLOAT, MPI_COMM_WORLD);

            for (int r=0; r<world_size; r++) if (max_values[r] > max_val) max_val = max_values[r];

            threshold_mpi(img_out2, first_row, last_row, max_val);

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
                for (int i = 1; i < height-1; i++)
                    for (int j = 1; j < width-1; j++) {
                        pixel2[i*width+j] = recv_buff[i*width+j];
                    }
                tracking(img_out2);
                t3 = omp_get_wtime();
                printf("  Done in %f\n", t3-t2);
                mpi_time += t3-t2;

                printf("Checking and saving results...\n");
                bool validate = true;
                for (int i = 0; i < height-1; i++)
                    for (int j = 0; j < width-1; j++) {
                        if (pixel1[i*width+j] != pixel2[i*width+j]) {
                            validate = false;
                        }
                    }
                if (validate) printf("  Results matched!\n\n");
                else printf("  Results unmatched!\n\n");

                img_out2.WriteToFile(outputname);
            }
        }
    }
    if (world_rank == 0) printf("MPI time: %f vs non-MPI time: %f\n", mpi_time/n_trials, non_mpi_time/n_trials);
    MPI_Finalize();
}

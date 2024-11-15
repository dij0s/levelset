#include <math.h>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include "../diagnostics/diagnostics.h"
#include "init.h"
#include "../common_includes.c"
using namespace std;

__global__ void vecAddKernel(double *distance, double *phi, double *u, double *v, int nx, int ny, double dx, double dy, double ycenter, double xcenter, double radius) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i < nx && j < ny) {

        double x_val = i * dx - xcenter;
        double y_val = j * dy - ycenter;
        
        distance[j * nx + i] = sqrt(x_val * x_val + y_val * y_val) - radius;

        phi[j * nx + i] = distance[j * nx + i];
        u[j * nx + i] = sin(2.0 * M_PI * j * dy) * sin(M_PI * i * dx) * sin(M_PI * i * dx);
        v[j * nx + i] = -sin(2.0 * M_PI * i * dx) * sin(M_PI * j * dy) * sin(M_PI * j * dy);
    }
}

// Initialization of the distance function inside the domain
void Initialization(double** phi, double** curvature, double** u, double** v, const int nx, const int ny, const double dx, const double dy) {
   /*
    // == Circle parameters ==
    double xcenter = 0.5; // Circle position x
    double ycenter = 0.75; // Circle position y
    double radius = 0.15; // Circle radius

    for (int i = 0; i < nx; i++){
        double x = i*dx - xcenter;    

        for (int j = 0; j < ny; j++){
            double y = j*dy - ycenter;

            // Compute the signed distance to the interface
            double distance = sqrt(x*x+y*y) - radius;

            phi[i][j] = distance;

            // Compute the velocity based on x and y
            u[i][j] = sin(2.0*M_PI*j*dy) * sin(M_PI*i*dx) * sin(M_PI*i*dx);
            v[i][j] = -sin(2.0*M_PI*i*dx) * sin(M_PI*j*dy) * sin(M_PI*j*dy);
            printf("%f\n",  phi[i][j]);
        }
    }
*/
    double xcenter = 0.5;
    double ycenter = 0.75;
    double radius = 0.15;

    double *h_distance, *h_phi, *h_u, *h_v;
    double  *d_distance, *d_phi, *d_u, *d_v;
    size_t size1d = nx * sizeof(double);
    size_t size2d = nx * ny * sizeof(double);


    h_distance = (double*)malloc(size1d);
    h_phi = (double*)malloc(size2d);
    h_u = (double*)malloc(size2d);
    h_v = (double*)malloc(size2d);


    cudaMalloc((void**)&d_distance, size2d);
    cudaMalloc((void**)&d_phi, size2d);
    cudaMalloc((void**)&d_u, size2d);
    cudaMalloc((void**)&d_v, size2d);

   
    cudaMemcpy(d_distance, h_distance, size2d, cudaMemcpyHostToDevice);

    dim3 blockDim(10, 10);
    dim3 gridDim((nx + blockDim.x - 1) / blockDim.x, (ny + blockDim.y - 1) / blockDim.y);
    vecAddKernel<<<gridDim, blockDim>>>(d_distance, d_phi, d_u, d_v, nx, ny, dx, dy, ycenter, xcenter, radius);
    cudaDeviceSynchronize();
    
    // Copy results back to host
    CHECK_ERROR(cudaMemcpy(h_phi, d_phi, size2d, cudaMemcpyDeviceToHost));
    cudaMemcpy(h_u, d_u, size2d, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_v, d_v, size2d, cudaMemcpyDeviceToHost);
   

    
    for(int i = 0; i< nx*ny; i++){
        int jj = i% nx; 
        int ii = floor(i/nx);
        phi[jj][ii] = h_phi[i];
        u[jj][ii] = h_u[i];
        v[jj][ii] = h_v[i];
    }

    // Free device memory
    cudaFree(d_distance);
    cudaFree(d_phi);
    cudaFree(d_u);
    cudaFree(d_v);

    // Free host temporary memory
    free(h_phi);
    free(h_u);
    free(h_v);
    free(h_distance);

    // Call the function to compute the interface curvature with updated phi
    computeInterfaceCurvature(phi, curvature, nx, ny, dx, dy);





}

#include <math.h>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include "../diagnostics/diagnostics.h"
#include "init.h"
#include "../common_includes.c"

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
void Initialization(double* phi, double* curvature, double* u, double* v, const int nx, const int ny, const double dx, const double dy, double * d_phi, double * d_distance, double * d_curvature, double * d_u, double * d_v) {
    double xcenter = 0.5;
    double ycenter = 0.75;
    double radius = 0.15;

    double *h_distance;
    size_t size1d = nx * sizeof(double);
    size_t size2d = nx * ny * sizeof(double);


    h_distance = (double*)malloc(size1d);


   
    cudaMemcpy(d_distance, h_distance, size2d, cudaMemcpyHostToDevice);

    dim3 blockDim(10, 10);
    dim3 gridDim((nx + blockDim.x - 1) / blockDim.x, (ny + blockDim.y - 1) / blockDim.y);
    vecAddKernel<<<gridDim, blockDim>>>(d_distance, d_phi, d_u, d_v, nx, ny, dx, dy, ycenter, xcenter, radius);
    cudaDeviceSynchronize();
    
    // Copy results back to host
    cudaMemcpy(phi, d_phi, size2d, cudaMemcpyDeviceToHost);
    cudaMemcpy(u, d_u, size2d, cudaMemcpyDeviceToHost);
    cudaMemcpy(v, d_v, size2d, cudaMemcpyDeviceToHost);

    // Free host temporary memory
    free(h_distance);

    // Call the function to compute the interface curvature with updated phi
    computeInterfaceCurvature(phi, curvature, nx, ny, dx, dy, d_phi, d_curvature);
}

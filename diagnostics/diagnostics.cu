#include <math.h>
#include <iostream>

#include "diagnostics.h"

using namespace std;

// This kernel computes the length (L) of a single cell
// it does so and writes it to its block-specific shared
// memory view before reducing it on global memory
__global__ void singleCellInterfaceLength(double *phi, double *partial_lengths, const int nx, const int ny, const double dx, const double dy, const int unidimensional_size, const double epsilon) {
    // compute unique thread index
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // don't handle non-existent indexes
    if (i >= unidimensional_size) {
        return;
    }

    // avoid computing for borders
    int ii = i % nx;
    int jj = floor((double)i / nx);

    if (ii == 0 || ii == nx - 1 || jj == 0 || jj == ny - 1) {
        partial_lengths[i] = 0.0;
        return;
    }

    // compute interface length
    double phi_x = (phi[i + 1] - phi[i - 1]) / 2.0 / dx; 
    double phi_y = (phi[i + (nx - 2)] - phi[i - (nx - 2)]) / 2.0 / dy; 
    // compute the norm of gradient: norm(grad(phi)) 
    double normGrad = sqrt(phi_x * phi_x + phi_y * phi_y);
    // compute the dirac function approximation
    double delta = (1.0 / sqrt(2.0 * M_PI * epsilon)) * exp( - (phi[i] * phi[i]) / (2.0 * epsilon));
    // L = delta * norm(grad(phi)) * dx * dy
    // put data in shared memory
    partial_lengths[i] = delta * normGrad * dx * dy;
}

__global__ void computeSingleCellCurvature(double *curvature, double *phi,const double dx, const double dy, const int nx, const int ny)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    // avoid computing for borders
    if (i > 0 && i < (nx - 1) && j > 0 && j < (ny - 1))
    {
        if (abs(phi[j * nx + i]) < 3.0 * dx)
        {
            int center = i + j * nx;
            int left = (i - 1) + j * nx;             // phi[i-1][j]
            int right = (i + 1) + j * nx;            // phi[i+1][j]
            int up = (j - 1) * nx + i;               // phi[i][j-1]
            int down = (j + 1) * nx + i;             // phi[i][j+1]
            int up_left = (i - 1) + (j - 1) * nx;    // phi[i-1][j-1]
            int up_right = (i + 1) + (j - 1) * nx;   // phi[i-1][j+1]
            int down_left = (i - 1) + (j + 1) * nx;  // phi[i+1][j-1]
            int down_right = (i + 1) + (j + 1) * nx; // phi[i+1][j+1]

            // Compute first derivatives
            double phi_x = (phi[right] - phi[left]) / 2.0 / dx;
            double phi_y = (phi[down] - phi[up]) / 2.0 / dy;

            // Compute second derivatives
            double phi_xx = (phi[right] - 2.0 * phi[center] + phi[left]) / dx / dx;
            double phi_yy = (phi[down] - 2.0 * phi[center] + phi[up]) / dy / dy;

            // Compute mixed derivative
            double phi_xy = -(phi[up_right] - phi[down_right] - phi[up_left] + phi[down_left]) / dx / dy / 4.0;
            curvature[j*nx+i] = (phi_xx * phi_y * phi_y - 2.0 * phi_x * phi_y * phi_xy + phi_yy * phi_x * phi_x) /
                                  pow(phi_x * phi_x + phi_y * phi_y, 1.5);
        }
        else
        {
            curvature[j*nx+i] = 0.0;
        }
}
}

// The total interface length (L) is computed by following the following algorithm
// L ~ sum_{i,j} delta(phi_{i,j}) norm (grad(phi)) dx dy
// with delta(phi) an approximation of the dirac function := delta(phi) = 1 / sqrt (2 * pi * epsilon) * exp (- phi*phi / 2 / epsilon) 
void computeInterfaceLength(double* phi, const int nx, const int ny, const double dx, const double dy){
    // Fixed parameter for the dirac function
    double epsilon = 0.001;
    
    // reduce phi to one dimension
    // only includes internal cells
    const int unidimensional_size = nx * ny;
    
    // allocate memory on the device
    // for host-scoped data
    const int N_THREADS = 1024;
    const int N_BLOCKS = ceil((double)(unidimensional_size)/N_THREADS);

    size_t unidimensional_size_bytes = unidimensional_size * sizeof(double);
    double *d_phi_n, *d_partial_lengths, *h_partial_lengths;
    // create host-scoped
    // individual blocks result
    h_partial_lengths = new double[unidimensional_size];

    cudaMalloc((void **)&d_phi_n, unidimensional_size_bytes);
    cudaMalloc((void **)&d_partial_lengths, unidimensional_size_bytes);

    // copy data to device memory
    cudaMemcpy(d_phi_n, phi, unidimensional_size_bytes, cudaMemcpyHostToDevice);

    // launch kernel with shared memory size
    singleCellInterfaceLength<<<N_BLOCKS, N_THREADS>>>(d_phi_n, d_partial_lengths, nx, ny, dx, dy, unidimensional_size, epsilon);
    cudaDeviceSynchronize();

    // copy block results from
    // device back to host
    cudaMemcpy(h_partial_lengths, d_partial_lengths, unidimensional_size_bytes, cudaMemcpyDeviceToHost);

    // final reduction on the host
    double length = 0.0;
    for (int i = 0; i < unidimensional_size; i++) {
        length += h_partial_lengths[i];
    }

    // free memory on device
	cudaFree(d_phi_n);
	cudaFree(d_partial_lengths);

    // deallocate memory
    delete[] h_partial_lengths;

    // Print the total interface length 
    cout << "The total interface length is " << length << " m\n";
}

// The interface curvature (L) is computed by following the following algorithm
// curvature = (phi_xx * phi_y **2 - 2.0 * phi_x * phi_y * phi_xy + phi_yy * phi_x **2) / (phi_x **2 + phi_y **2) ** (3/2)
// with phi_x:= d phi / dx, phi_y:= d phi / dy
// and phi_xx:= d phi_x / dx, phi_yy:= d phi_y / dy, phi_xy:= d phi_x / dy
void computeInterfaceCurvature(double *phi, double *curvature, const int nx, const int ny, const double dx, const double dy)
{
    double maxCurvature = 0.0;

    double *d_curvature, *d_phi; 
    int size2d = nx * ny;
    size_t size2d_bytes = size2d * sizeof(double);

    cudaMalloc((void**)&d_curvature, size2d_bytes);
    cudaMalloc((void**)&d_phi, size2d_bytes);

    cudaMemcpy(d_phi, phi, size2d_bytes, cudaMemcpyHostToDevice);

    dim3 blockDim(10, 10);
    dim3 gridDim((nx + blockDim.x - 1) / blockDim.x, (ny + blockDim.y - 1) / blockDim.y);
    computeSingleCellCurvature<<<gridDim, blockDim>>>(d_curvature, d_phi, dx, dy, nx, ny);
    cudaDeviceSynchronize();
    
    cudaMemcpy(curvature, d_curvature, size2d_bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_curvature);
    cudaFree(d_phi);

    for (int i = 0; i < size2d; i++) {
        if (abs(curvature[i]) > maxCurvature)
        {
            maxCurvature = abs(curvature[i]);
        }
    }

    cout << "The maximum curvature is " << maxCurvature << " m^{-2}\n";
}

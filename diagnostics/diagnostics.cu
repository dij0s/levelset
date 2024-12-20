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
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    // don't handle non-existent indexes
    if (i >= unidimensional_size) {
        return;
    }

    // avoid computing for borders
    if (i == 0 || i == nx - 1 || j == 0 || j == ny - 1) {
        partial_lengths[j * nx + i] = 0.0;
        return;
    }

    // compute interface length
    int left = (i - 1) + j * nx;             // phi[i-1][j]
    int right = (i + 1) + j * nx;            // phi[i+1][j]
    int up = (j - 1) * nx + i;               // phi[i][j-1]
    int down = (j + 1) * nx + i;             // phi[i][j+1]
    double phi_x = (phi[right] - phi[left]) / 2.0 / dx;
    double phi_y = (phi[down] - phi[up]) / 2.0 / dy;
    // compute the norm of gradient: norm(grad(phi)) 
    double normGrad = sqrt(phi_x * phi_x + phi_y * phi_y);
    // compute the dirac function approximation
    double delta = (1.0 / sqrt(2.0 * M_PI * epsilon)) * exp( - (phi[j * nx + i] * phi[j * nx + i]) / (2.0 * epsilon));
    // L = delta * norm(grad(phi)) * dx * dy
    // put data in shared memory
    partial_lengths[j * nx + i] = delta * normGrad * dx * dy;
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
void computeInterfaceLength(double* phi, const int nx, const int ny, const double dx, const double dy,double* d_phi, double* d_phi_n, double* d_partial_lengths){
    // Fixed parameter for the dirac function
    double epsilon = 0.001;
    
    // reduce phi to one dimension
    // only includes internal cells
    const int unidimensional_size = nx * ny;
    
    // allocate memory on the device
    // for host-scoped data
    dim3 blockDim(10, 10);
    dim3 gridDim((nx + blockDim.x - 1) / blockDim.x, (ny + blockDim.y - 1) / blockDim.y);

    size_t unidimensional_size_bytes = unidimensional_size * sizeof(double);
    // create host-scoped
    // individual blocks result
    double *h_partial_lengths;
    h_partial_lengths = new double[unidimensional_size];

    // launch kernel with shared memory size
    singleCellInterfaceLength<<<gridDim, blockDim>>>(d_phi_n, d_partial_lengths, nx, ny, dx, dy, unidimensional_size, epsilon);
    cudaDeviceSynchronize();

    // copy block results from
    // device back to host
    cudaMemcpy(h_partial_lengths, d_partial_lengths, unidimensional_size_bytes, cudaMemcpyDeviceToHost);

    // final reduction on the host
    double length = 0.0;
    for (int i = 0; i < unidimensional_size; i++) {
        length += h_partial_lengths[i];
    }



    // deallocate memory
    delete[] h_partial_lengths;

    // Print the total interface length 
    cout << "The total interface length is " << length << " m\n";
}

// The interface curvature (L) is computed by following the following algorithm
// curvature = (phi_xx * phi_y **2 - 2.0 * phi_x * phi_y * phi_xy + phi_yy * phi_x **2) / (phi_x **2 + phi_y **2) ** (3/2)
// with phi_x:= d phi / dx, phi_y:= d phi / dy
// and phi_xx:= d phi_x / dx, phi_yy:= d phi_y / dy, phi_xy:= d phi_x / dy
void computeInterfaceCurvature(double *phi, double *curvature, const int nx, const int ny, const double dx, const double dy, double* d_phi, double* d_curvature)
{
    double maxCurvature = 0.0;

    int size2d = nx * ny;
    size_t size2d_bytes = size2d * sizeof(double);

    dim3 blockDim(10, 10);
    dim3 gridDim((nx + blockDim.x - 1) / blockDim.x, (ny + blockDim.y - 1) / blockDim.y);
    computeSingleCellCurvature<<<gridDim, blockDim>>>(d_curvature, d_phi, dx, dy, nx, ny);
    cudaDeviceSynchronize();
    
    cudaMemcpy(curvature, d_curvature, size2d_bytes, cudaMemcpyDeviceToHost);

    for (int i = 0; i < size2d; i++) {
        if (abs(curvature[i]) > maxCurvature)
        {
            maxCurvature = abs(curvature[i]);
        }
    }

    cout << "The maximum curvature is " << maxCurvature << " m^{-2}\n";
}

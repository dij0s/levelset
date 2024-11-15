#include <math.h>
#include <iostream>
#include <cooperative_groups.h>

#include "diagnostics.h"

using namespace std;
namespace cg = cooperative_groups;

// This kernel computes the length (L) of a single cell
// it does so and writes it to its block-specific shared
// memory view before reducing it on global memory
__global__ void singleCellInterfaceLength(double *phi, double *block_results, const int nx, const int ny, const double dx, const double dy, const int unidimensional_size, const double epsilon) {
    // handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    // dynamically allocated shared memory
    // its size is given when running the
    // kernel
    extern __shared__ double sdata[];

    // compute unique thread index
	int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // don't handle non-existent indexes
    if (i >= unidimensional_size) {
        sdata[tid] = 0;
    }

    // compute interface length
    double phi_x = (phi[i + 1] - phi[i - 1]) / 2.0 / dx; 
    double phi_y = (phi[i + nx] - phi[i - nx]) / 2.0 / dy; 
    // compute the norm of gradient: norm(grad(phi)) 
    double normGrad = sqrt(phi_x * phi_x + phi_y * phi_y);
    // compute the dirac function approximation
    double delta = (1.0 / sqrt(2.0 * M_PI * epsilon)) * exp( - (phi[i] * phi[i]) / (2.0 * epsilon));
    // L = delta * norm(grad(phi)) * dx * dy
    // put data in shared memory
    sdata[tid] = delta * normGrad * dx * dy;

    // synchronize all threads in block
    // to ensure they have all computed
    // and stored their length in shared
    // memory
    cg::sync(cta);

    // do reduction in shared memory
    // it is done in log_2(n) operations
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        cg::sync(cta);
    }

    // write the block result to global memory
    if (tid == 0) block_results[blockIdx.x] = sdata[0];
}

// The total interface length (L) is computed by following the following algorithm
// L ~ sum_{i,j} delta(phi_{i,j}) norm (grad(phi)) dx dy
// with delta(phi) an approximation of the dirac function := delta(phi) = 1 / sqrt (2 * pi * epsilon) * exp (- phi*phi / 2 / epsilon) 
void computeInterfaceLength(double** phi, const int nx, const int ny, const double dx, const double dy){
    // Fixed parameter for the dirac function
    double epsilon = 0.001;
    
    // reduce phi to one dimension
    const int unidimensional_size = nx * ny;
    double* phi_n = new double[unidimensional_size];
    
    for (int i = 0; i < unidimensional_size; i++) {
        // compute two dimensional index
        int ii = i % nx;
        int jj = floor(i / nx);

        // assign value to copy of phi
        phi_n[i] = phi[ii][jj];
    }

    // allocate memory on the device
    // for host-scoped data
    const int N_THREADS = 1024;
    const int N_BLOCKS = ceil((double)(unidimensional_size)/N_THREADS);

    size_t unidimensional_size_bytes = unidimensional_size * sizeof(double);
    double *d_phi_n, *h_block_results, *d_block_results;
    // create host-scoped
    // individual blocks result
    h_block_results = new double[N_BLOCKS];

    cudaMalloc((void **)&d_phi_n, unidimensional_size_bytes);
    cudaMalloc((void **)&d_block_results, N_BLOCKS * sizeof(double));

    // copy data to device memory
    cudaMemcpy(d_phi_n, phi_n, unidimensional_size_bytes, cudaMemcpyHostToDevice);

    // launch kernel with shared memory size
    size_t shared_memory_size = N_THREADS * sizeof(double);
    singleCellInterfaceLength<<<N_BLOCKS, N_THREADS, shared_memory_size>>>(d_phi_n, d_block_results, nx, ny, dx, dy, unidimensional_size, epsilon);
    cudaDeviceSynchronize();

    // copy block results from
    // device back to host
    cudaMemcpy(h_block_results, d_block_results, N_BLOCKS * sizeof(double), cudaMemcpyDeviceToHost);

    // final reduction on the host
    double length = 0.0;
    for (int i = 0; i < N_BLOCKS; i++) {
        length += h_block_results[i];
    }

    // free memory on device
	cudaFree(d_phi_n);
	cudaFree(d_block_results);

    // deallocate memory
    delete[] phi_n;
    delete[] h_block_results;

    // Print the total interface length 
    cout << "The total interface length is " << length << " m\n";
}


// The interface curvature (L) is computed by following the following algorithm
// curvature = (phi_xx * phi_y **2 - 2.0 * phi_x * phi_y * phi_xy + phi_yy * phi_x **2) / (phi_x **2 + phi_y **2) ** (3/2) 
// with phi_x:= d phi / dx, phi_y:= d phi / dy 
// and phi_xx:= d phi_x / dx, phi_yy:= d phi_y / dy, phi_xy:= d phi_x / dy 
void computeInterfaceCurvature(double** phi,double** curvature, const int nx, const int ny, const double dx, const double dy){

    double maxCurvature = 0.0;

    for (int i = 1; i < nx-1; i++){
        for (int j = 1; j < ny-1; j++){

            if (abs(phi[i][j]) < 3.0 * dx ){ //Compute the curvature only near the interface 

                // first derivative
                double phi_x = (phi[i+1][j]-phi[i-1][j])/2.0/dx;
                double phi_y = (phi[i][j+1]-phi[i][j-1])/2.0/dy;

                // second derivative
                double phi_xx = (phi[i+1][j]-2.0*phi[i][j]+phi[i-1][j])/dx/dx;
                double phi_yy = (phi[i][j+1]-2.0*phi[i][j]+phi[i][j-1])/dy/dy;
                double phi_xy = (phi[i+1][j+1] - phi[i+1][j-1] - phi[i-1][j+1] + phi[i-1][j-1])/dx/dy/4.0;

                // compute curvature
                curvature[i][j] = (phi_xx*phi_y*phi_y - 2.0* phi_x*phi_y*phi_xy + phi_yy*phi_x*phi_x)/
                    pow(phi_x*phi_x+phi_y*phi_y,1.5);

                // Replace the maximum curvature
                if (abs(curvature[i][j]) > maxCurvature){maxCurvature = abs(curvature[i][j]);}

            }
            else {// Default value if the cell is not closed to the interface 
                curvature[i][j] = 0.0;
            }
        }
    }

    // Print the maximum interface curvature 
    cout << "The maximum curvature is " << maxCurvature << " m^{-2}\n";
}
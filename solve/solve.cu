#include "solve.h"

#include <cuda.h>
#include <cmath>

using namespace std;

// This kernel computes φ(t+Δt) of a single cell
// based on its horizontal speed u, vertical speed
// v and its cardinal neighbors
__global__ void singleCellEquationExplicit(double *phi, double *phi_n, double *u, double *v, const double dt, const double dx, const double dy, const int nxy, const int size) {
    // compute unique thread index
	int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // don't handle non-existent indexes
    // due to 2d ceiling necessity
    if (i > size) {
        return;
    }

    // compute two-dimensional index
    // for horizontal and vertical
    // boundary checking -> don't handle
    // if it is on boundary
    int ii = i % nxy;
    if (ii == 0 || ii == (nxy - 1)) {
        return;
    }
    int jj = floor((double)i / nxy);
    if (jj == 0 || jj == (nxy - 1)) {
        return;
    }

    // compute φ(t+Δt)
    phi[i] = phi_n[i];
    
    if (u[i] < 0.0) {
        phi[i] -= dt * (u[i]*(phi_n[i+1] - phi_n[i])/dx);
    }
    else {
        phi[i] -= dt * (u[i]*(phi_n[i] - phi_n[i-1])/dx);
    }

    if (v[i] < 0.0) {
        phi[i] -= dt * (v[i]*(phi_n[i+nxy] - phi_n[i])/dy);
    }
    else {
        phi[i] -= dt * (v[i]*(phi_n[i] - phi_n[i-nxy])/dy);
    } 
}

// Compute the boundaries of the domain for the phi field
void computeBoundaries(double** phi, const int nx, const int ny){
    // Upper and Lower boundaries (extrapolation)
    for (int i = 0; i < nx ; i++){
        phi[i][ny-1] = 2.0 * phi[i][ny-2] - phi[i][ny-3];
        phi[i][0] = 2.0 * phi[i][1] - phi[i][2];
    }

    // Left and Right boundaries (extrapolation)
    for (int j = 0; j < ny; j++){
        phi[0][j] = 2.0 * phi[1][j] - phi[2][j];
        phi[nx-1][j] = 2.0 * phi[nx-2][j] - phi[nx-3][j];
    }
}


// Solving advection equation on the inside domain ([1;nx-2] x [1;ny-2])
// Equation solved: d phi / dt + u d phi / dx + v d phi / dy = 0
// Using the euler explicit numerical scheme => phi = phi_n - (u d phi / dx + v d phi / dy)
// A first order upwind scheme is used to stabilize the solver (https://en.wikipedia.org/wiki/Upwind_scheme)
void solveAdvectionEquationExplicit(
    double** phi, double** u, double** v, const int nx, const int ny, const double dx, const double dy, const double dt){

    // Copy phi to phi_n
    // phi is reduced to a
    // unidimentional datastructure
    const int unidimensional_size = nx * ny;
    
    double* phi_n = new double[unidimensional_size];
    double* single_dimension_u = new double[unidimensional_size];
    double* single_dimension_v = new double[unidimensional_size];
    
    for (int i = 0; i < unidimensional_size; i++) {
        // compute two dimensional index
        int ii = i % nx;
        int jj = floor(i / nx);

        // assign value to copy of phi
        // and dimension-reduced u and v
        phi_n[i] = phi[ii][jj];
        single_dimension_u[i] = u[ii][jj];
        single_dimension_v[i] = v[ii][jj];
    }

    // Compute the advection equation 
    
    // allocate memory on the device
    // for host-scoped data
    size_t unidimensional_size_bytes = unidimensional_size * sizeof(double);
    double *d_phi, *d_phi_n, *d_u, *d_v;

    cudaMalloc((void **)&d_phi, unidimensional_size_bytes);
    cudaMalloc((void **)&d_phi_n, unidimensional_size_bytes);
    cudaMalloc((void **)&d_u, unidimensional_size_bytes);
    cudaMalloc((void **)&d_v, unidimensional_size_bytes);

    // copy data to device memory
    cudaMemcpy(d_phi_n, phi_n, unidimensional_size_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_u, single_dimension_u, unidimensional_size_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, single_dimension_v, unidimensional_size_bytes, cudaMemcpyHostToDevice);

    const int N_THREADS = 1024;
    const int N_BLOCKS = ceil((double)(unidimensional_size)/N_THREADS);

	singleCellEquationExplicit<<<N_BLOCKS, N_THREADS>>>(d_phi, d_phi_n, d_u, d_v, dt, dx, dy, nx, unidimensional_size);
    cudaDeviceSynchronize();

    // copy result from device back to host
	cudaMemcpy(phi_n, d_phi, unidimensional_size_bytes, cudaMemcpyDeviceToHost);

    // free memory on device
	cudaFree(d_phi);
	cudaFree(d_phi_n);
	cudaFree(d_u);
	cudaFree(d_v);

    // transform back to 2d array
    for (int i = 0; i < unidimensional_size; i++) {
        int ii = i % nx;
        int jj = floor(i / nx);
        
        phi[ii][jj] = phi_n[i];
    }

    // Refresh the boundaries values
    computeBoundaries(phi, nx, ny);

    // Deallocate memory
    delete[] phi_n;
    delete[] single_dimension_u;
    delete[] single_dimension_v;
}
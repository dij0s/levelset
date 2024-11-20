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

// This kernel computes and updates
// the boundaries of phi
__global__ void singleCellBoundaries(double *phi, const int nx, const int ny) {
    // compute unique thread index
	int i = blockIdx.x * blockDim.x + threadIdx.x;

    // compute 2d-equivalent index
    int ii = i % nx;
    int jj = floor((double)i / nx);

    // only handle cell if it is on boundary
    // extrapolate to compute its value
    if (ii == 0) {
        phi[i] = 2.0 * phi[i + nx] - phi[i + 2 * nx];
    } else if (ii == nx - 1) {
        phi[i] = 2.0 * phi[i - ny] - phi[i - 2 * ny];
    } else if (jj == 0) {
        phi[i] = 2.0 * phi[i + 1] - phi[i + 2];
    } else if (jj = ny - 1) {
        phi[i] = 2.0 * phi[i - 1] - phi[i - 2];
    } else {
        return;
    }
}

// Compute the boundaries of the domain for the phi field
void computeBoundaries(double* phi, const int nx, const int ny){
    int unidimensional_size = nx * ny;

    const int N_THREADS = 1024;
    const int N_BLOCKS = ceil((double)(unidimensional_size)/N_THREADS);

    singleCellBoundaries<<<N_THREADS, N_BLOCKS>>>(phi, nx, ny);
    cudaDeviceSynchronize();
}


// Solving advection equation on the inside domain ([1;nx-2] x [1;ny-2])
// Equation solved: d phi / dt + u d phi / dx + v d phi / dy = 0
// Using the euler explicit numerical scheme => phi = phi_n - (u d phi / dx + v d phi / dy)
// A first order upwind scheme is used to stabilize the solver (https://en.wikipedia.org/wiki/Upwind_scheme)
void solveAdvectionEquationExplicit(
    double *d_phi, double *d_phi_n, double *d_u, double *d_v, const int nx, const int ny, const int dx, const int dy, const double dt){

    const int unidimensional_size = nx * ny;
    size_t unidimensional_size_bytes = nx * ny * sizeof(double);
    double* phi_n = new double[unidimensional_size];
    
    cudaMemcpy(phi_n, d_phi, unidimensional_size_bytes, cudaMemcpyDeviceToHost);

    // Compute the advection equation 
    
    // allocate memory on the device
    // for host-scoped data
    cudaMemcpy(d_phi_n, phi_n, unidimensional_size_bytes, cudaMemcpyHostToDevice);

    const int N_THREADS = 1024;
    const int N_BLOCKS = ceil((double)(unidimensional_size)/N_THREADS);

	singleCellEquationExplicit<<<N_BLOCKS, N_THREADS>>>(d_phi, d_phi_n, d_u, d_v, dt, dx, dy, nx, unidimensional_size);
    cudaDeviceSynchronize();

    // Refresh the boundaries values
    computeBoundaries(d_phi, nx, ny);

    // Deallocate memory
    delete[] phi_n;
}
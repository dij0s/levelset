// Libraries
#include <iostream>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <chrono>
//#include <functional>
#include <cuda.h>
#include <thread>
// == User lib ==
#include "diagnostics/diagnostics.h"
#include "initialization/init.h"
#include "solve/solve.h"
#include "write/write.h"

// Namespace
using namespace std;
using namespace chrono;

// Advection Solver 
int main(int argc, char *argv[])
{
    auto start_total = high_resolution_clock::now();

    // Data Initialization
    // == Spatial ==
    int scale = 1; 
    if (argc > 1){scale = stoi(argv[1]);}
    
    int nx = 100*scale; int ny = 100*scale; // Number of cells in each direction 
    double Lx = 1.0; double Ly = 1.0; // Square domain [m]
    double dx = Lx / (nx-1); double dy = Ly / (ny-1); // Spatial step [m]
    
    // == Temporal ==
    double tFinal = 2.0; // Final time [s]
    double dt = 0.005/scale; // Temporal step [s]
    int nSteps = int(tFinal/dt); // Number of steps to perform
    double time = 0.0; // Actual Simulation time [s]

    // == Numerical == 
    int outputFrequency = nSteps/40;
    
    const int unidimensional_size = nx * ny;
    size_t unidimensional_size_bytes = unidimensional_size* sizeof(double);
    
    double* phi = (double*) malloc(unidimensional_size_bytes); // LevelSet field
    double* curvature = (double*) malloc(unidimensional_size_bytes); // Curvature field
    double* u = (double*) malloc(unidimensional_size_bytes); // Velocity field in x-direction
    double* v = (double*) malloc(unidimensional_size_bytes); // Velocity field in y-direction

    auto duration_init = duration_cast<nanoseconds>(high_resolution_clock::now() - start_total);
    printf("Execution time of non-p initialization: %ldns\n", duration_init.count());
    
    double *d_distance, *d_phi, *d_u, *d_v, *d_phi_n, *d_partial_lengths, *d_curvature;

    cudaMalloc((void**)&d_distance, unidimensional_size_bytes);
    cudaMalloc((void**)&d_phi, unidimensional_size_bytes);
    cudaMalloc((void**)&d_u, unidimensional_size_bytes);
    cudaMalloc((void**)&d_v, unidimensional_size_bytes);
    cudaMalloc((void **)&d_phi_n, unidimensional_size_bytes);
    cudaMalloc((void **)&d_partial_lengths, unidimensional_size_bytes);
    cudaMalloc((void **)&d_curvature, unidimensional_size_bytes);
   
    Initialization(d_phi, d_distance, d_curvature, d_u, d_v, nx, ny, dx, dy); // Initialize the distance function field 
    computeBoundaries(d_phi, nx, ny); // Extrapolate phi on the boundaries
    // == Output ==
    auto start_write = high_resolution_clock::now();
    
    stringstream ss;
    ss << scale;
    string scaleStr = ss.str();

    string outputName =  "output/levelSet_scale"+scaleStr+"_";
    int count = 0; // Number of VTK file already written 

    // == First output == 
    // Write data in VTK format
    mkdir("output", 0777); // Create output folder

    auto duration_write = duration_cast<nanoseconds>(high_resolution_clock::now() - start_write);
    printf("Execution time of non-p writing: %ldns\n", duration_write.count());

    // copy data back from device
    // to write the vtk file
    cudaMemcpy(phi, d_phi, unidimensional_size_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(curvature, d_curvature, unidimensional_size_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(u, d_u, unidimensional_size_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(v, d_v, unidimensional_size_bytes, cudaMemcpyDeviceToHost);

    writeDataVTK(outputName, phi, curvature, u, v, nx, ny, dx, dy, count++);
    
    // Loop over time
    for (int step = 1; step <= nSteps; step++){

        time += dt; // Simulation time increases
        cout << "\nStarting iteration step " << step << "/"<< nSteps << "\tTime " << time << "s\n"; 

        // Solve the advection equation
        solveAdvectionEquationExplicit(d_phi, d_phi_n, d_u, d_v, nx, ny, dx, dy, dt);

        // Diagnostics: interface perimeter
        computeInterfaceLength(d_phi, d_partial_lengths, nx, ny, dx, dy);

        // Diagnostics: interface curvature
        computeInterfaceCurvature(d_phi, d_curvature, nx, ny, dx, dy);
        
        // Write data to output file
        // Deallocate GPU memory
        if (step%outputFrequency == 0){
            // copy data into array copies
            cudaMemcpy(phi, d_phi, unidimensional_size_bytes, cudaMemcpyDeviceToHost);
            cudaMemcpy(curvature, d_curvature, unidimensional_size_bytes, cudaMemcpyDeviceToHost);
            cudaMemcpy(u, d_u, unidimensional_size_bytes, cudaMemcpyDeviceToHost);
            cudaMemcpy(v, d_v, unidimensional_size_bytes, cudaMemcpyDeviceToHost);
            
            thread newThread((writeDataVTK), outputName, phi, curvature, u, v, nx, ny, dx, dy, count++);
            newThread.detach();
        }
    }

    auto start_deallocate = high_resolution_clock::now();

    cudaFree(d_distance);
    cudaFree(d_phi);
    cudaFree(d_u);
    cudaFree(d_v);
    cudaFree(d_phi_n);
    cudaFree(d_partial_lengths);
    cudaFree(d_curvature);

    // Deallocate memory
    delete[] phi;
    delete[] curvature;
    delete[] u;
    delete[] v;

    auto duration_deallocate = duration_cast<nanoseconds>(high_resolution_clock::now() - start_deallocate);
    printf("Execution time of memory deallocation: %ldns\n", duration_deallocate.count());

    auto duration_total = duration_cast<milliseconds>(high_resolution_clock::now() - start_total);
    printf("Total execution time for %fs of simulation: %ldms\n", tFinal, duration_total.count());

    // must wait for writing
    // thread to finish
    return 0;
}

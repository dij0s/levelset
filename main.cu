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
    cout << "Hello\n";
    auto start_total = high_resolution_clock::now();

    // Data Initialization
    // == Spatial ==
    int scale = 1; 
    if (argc > 1){scale = stoi(argv[1]);}
    printf("scale: %d",scale);
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
    
    size_t unidimensional_size_bytes = nx * ny * sizeof(double);
    
    double* phi = (double*) malloc(unidimensional_size_bytes); // LevelSet field
    double* curvature = (double*) malloc(unidimensional_size_bytes); // Curvature field
    double* u = (double*) malloc(unidimensional_size_bytes); // Velocity field in x-direction
    double* v = (double*) malloc(unidimensional_size_bytes); // Velocity field in y-direction

    auto duration_init = duration_cast<nanoseconds>(high_resolution_clock::now() - start_total);
    printf("Execution time of non-p initialization: %ldns\n", duration_init.count());
    

    double  *d_distance, *d_phi, *d_u, *d_v, *d_phi_n, *d_partial_lengths, *d_curvature;

    cudaMalloc((void**)&d_distance, unidimensional_size_bytes);
    cudaMalloc((void**)&d_phi, unidimensional_size_bytes);
    cudaMalloc((void**)&d_u, unidimensional_size_bytes);
    cudaMalloc((void**)&d_v, unidimensional_size_bytes);
    cudaMalloc((void **)&d_phi_n, unidimensional_size_bytes);
    cudaMalloc((void **)&d_partial_lengths, unidimensional_size_bytes);
    cudaMalloc((void **)&d_curvature, unidimensional_size_bytes);

   
    Initialization(phi, curvature, u, v, nx, ny, dx, dy, d_phi, d_distance, d_curvature, d_u, d_v); // Initialize the distance function field 
    computeBoundaries(phi, nx, ny); // Extrapolate phi on the boundaries
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

    writeDataVTK(outputName, phi, curvature, u, v, nx, ny, dx, dy, count++);

    // setup array copies
    const int unidimensional_size = nx * ny;
    
    double* phi_copy = new double[unidimensional_size];
    double* curvature_copy = new double[unidimensional_size];
    double* u_copy = new double[unidimensional_size];
    double* v_copy = new double[unidimensional_size];
    // Loop over time
    for (int step = 1; step <= nSteps; step++){

        time += dt; // Simulation time increases
        cout << "\nStarting iteration step " << step << "/"<< nSteps << "\tTime " << time << "s\n"; 

        // Solve the advection equation
        solveAdvectionEquationExplicit(phi, u, v, nx, ny, dx, dy, dt, d_phi, d_phi_n, d_u,d_v);

        // Diagnostics: interface perimeter
        computeInterfaceLength(phi, nx, ny, dx, dy, d_phi, d_phi_n, d_partial_lengths);

        // Diagnostics: interface curvature
        computeInterfaceCurvature(phi, curvature, nx, ny, dx, dy, d_phi, d_curvature);
        
        // Write data to output file
        // Deallocate GPU memory
        if (step%outputFrequency == 0){
            // copy data into array copies
            for (int i = 0; i < unidimensional_size; i++) {
                phi_copy[i] = phi[i];
                curvature_copy[i] = curvature[i];
                u_copy[i] = u[i];
                v_copy[i] = v[i];
            }
            
            thread newThread((writeDataVTK), outputName, phi_copy, curvature_copy, u_copy, v_copy, nx, ny, dx, dy, count++);
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

    delete[] phi_copy;
    delete[] curvature_copy;
    delete[] u_copy;
    delete[] v_copy;
   
    auto duration_deallocate = duration_cast<nanoseconds>(high_resolution_clock::now() - start_deallocate);
    printf("Execution time of memory deallocation: %ldns\n", duration_deallocate.count());

    auto duration_total = duration_cast<milliseconds>(high_resolution_clock::now() - start_total);
    printf("Total execution time for %fs of simulation: %ldms\n", tFinal, duration_total.count());

    // must wait for writing
    // thread to finish
    return 0;
}

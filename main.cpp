// Libraries
#include <iostream>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <chrono>
#include <thread>
#include <functional>

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
    double** phi = new double*[nx]; // LevelSet field
    double** curvature = new double*[nx]; // Curvature field
    double** u = new double*[nx]; // Velocity field in x-direction
    double** v = new double*[nx]; // Velocity field in y-direction
    for (int i = 0; i < nx; ++i) {
        phi[i] = new double[ny];
        curvature[i] = new double[ny];
        u[i] = new double[ny];
        v[i] = new double[ny];
    }

    auto duration_init = duration_cast<nanoseconds>(high_resolution_clock::now() - start_total);
    printf("Execution time of non-p initialization: %ldns\n", duration_init.count());

    Initialization(phi, curvature, u, v, nx, ny, dx, dy); // Initialize the distance function field 
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

    // Loop over time
    for (int step = 1; step <= nSteps; step++){

        time += dt; // Simulation time increases
        cout << "\nStarting iteration step " << step << "/"<< nSteps << "\tTime " << time << "s\n"; 

        // Solve the advection equation
        solveAdvectionEquationExplicit(phi, u, v, nx, ny, dx, dy, dt);

        // Diagnostics: interface perimeter
        computeInterfaceLength(phi, nx, ny, dx, dy);

        // Diagnostics: interface curvature
        computeInterfaceCurvature(phi, curvature, nx, ny, dx, dy);

        // Write data to output file
        if (step%outputFrequency == 0){

            thread newThread(function<void(string, double**, double**, double**, double**, int, int, double, double, int)>(writeDataVTK), outputName, phi, curvature, u, v, nx, ny, dx, dy, count++);
            newThread.detach();
            //writeDataVTK(outputName, phi, curvature, u, v, nx, ny, dx, dy, count++);
        }

    }

    auto start_deallocate = high_resolution_clock::now();

    // Deallocate memory
    for (int i = 0; i < nx; ++i) {
        delete[] phi[i];
    }
    delete[] phi;

    auto duration_deallocate = duration_cast<nanoseconds>(high_resolution_clock::now() - start_deallocate);
    printf("Execution time of memory deallocation: %ldns\n", duration_deallocate.count());

    auto duration_total = duration_cast<milliseconds>(high_resolution_clock::now() - start_total);
    printf("Total execution time for %fs of simulation: %ldms\n", tFinal, duration_total.count());

    return 0;
}

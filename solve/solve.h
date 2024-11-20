#ifndef SOLVE_H
#define SOLVE_H

using namespace std;

void computeBoundaries(double* phi, const int nx, const int ny);
void solveAdvectionEquationExplicit(
    double* phi, double* u, double* v, const int nx, const int ny, const double dx, const double dy, const double dt, double* d_phi, double* d_phi_n, double* d_u, double* d_v);

#endif // SOLVE_H

#ifndef SOLVE_H
#define SOLVE_H

using namespace std;

void computeBoundaries(double* phi, const int nx, const int ny);
void solveAdvectionEquationExplicit(
    double *d_phi, double *d_phi_n, double *d_u, double *d_v, const int nx, const int ny, const int dx, const int dy, const double dt);

#endif // SOLVE_H

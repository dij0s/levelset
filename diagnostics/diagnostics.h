#ifndef DIAGNOSTICS_H
#define DIAGNOSTICS_H

using namespace std;

void computeInterfaceLength(double* phi, const int nx, const int ny, const double dx, const double dy, double* d_phi, double* d_phi_n, double* d_partial_lengths);
void computeInterfaceCurvature(double* phi,double* curvature, const int nx, const int ny, const double dx, const double dy, double* d_phi, double* d_curvature);

#endif // DIAGNOSTICS_H

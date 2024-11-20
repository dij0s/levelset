#ifndef DIAGNOSTICS_H
#define DIAGNOSTICS_H

using namespace std;

void computeInterfaceLength(double *d_phi, double *d_partial_lengths, const int nx, const int ny, const int dx, const int dy);
void computeInterfaceCurvature(double* d_phi, double* d_curvature, const int nx, const int ny, const double dx, const double dy);

#endif // DIAGNOSTICS_H

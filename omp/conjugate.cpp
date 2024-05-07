#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>
#include <fstream>

bool conjugateGradient(const std::vector<std::vector<double>>& A,
                       const std::vector<double>& b,
                       std::vector<double>& x,
                       double tol = 1e-6,
                       int maxIter = 1000) {

    //std::ofstream outFile("convergence.txt");

    int n = b.size();
    std::vector<double> r(n), p(n), Ap(n);
    double alpha, beta, rsold, rsnew;

    // Calculate initial residual
#pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        r[i] = b[i];
        for (int j = 0; j < n; ++j) {
            r[i] -= A[i][j] * x[j];
        }
        p[i] = r[i];
    }

    rsold = 0.0;
#pragma omp parallel for reduction(+:rsold)
    for (int i = 0; i < n; ++i) {
        rsold += r[i] * r[i];
    }

    for (int iter = 0; iter < maxIter; ++iter) {
        // Compute Ap = A * p
#pragma omp parallel for
        for (int i = 0; i < n; ++i) {
            Ap[i] = 0.0;
            for (int j = 0; j < n; ++j) {
                Ap[i] += A[i][j] * p[j];
            }
        }

        // Compute alpha
        double pAp = 0.0;
	double rp = 0.0;
#pragma omp parallel for reduction(+:pAp)
        for (int i = 0; i < n; ++i) {
            pAp += p[i] * Ap[i];
	    rp += p[i]*r[i];
        }
        alpha = rsold / pAp;

        // Update solution and residual
#pragma omp parallel for
        for (int i = 0; i < n; ++i) {
            x[i] += alpha * p[i];
            r[i] -= alpha * Ap[i];
        }

        // Compute squared norm of new residual
        rsnew = 0.0;
#pragma omp parallel for reduction(+:rsnew)
        for (int i = 0; i < n; ++i) {
            rsnew += r[i] * r[i];
        }

        // Check for convergence
        if (std::sqrt(rsnew) < tol) {
	    //std::cout << "Iterations for Convergence : " << iter+1 << std::endl;
            return true;
        }

        // Compute beta
        beta = rsnew / rsold;

        // Update search direction
#pragma omp parallel for
        for (int i = 0; i < n; ++i) {
            p[i] = r[i] + beta * p[i];
        }

        rsold = rsnew;

	//Log the convergence
	//outFile << iter+1 << " " << std::sqrt(rsnew) << std::endl;
    }

    //Close the file
    //outFile.close();

    // Failed to converge within maxIter iterations
    return false;
}


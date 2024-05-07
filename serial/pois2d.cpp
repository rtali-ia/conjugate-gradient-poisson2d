#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>
#include <fstream>
#include <chrono>

bool conjugateGradient(const std::vector<std::vector<double>>& A,
                       const std::vector<double>& b,
                       std::vector<double>& x,
                       double tol,
                       int maxIter);

// Function to write the solution vector to a file
void writeSolutionToFile(const std::vector<double>& x, int N) {
    std::ofstream outputFile("solution.txt");
    if (outputFile.is_open()) {
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                outputFile << x[i * N + j] << " ";
            }
            outputFile << std::endl;
        }
        outputFile.close();
    } else {
        std::cerr << "Error: Unable to open file for writing." << std::endl;
    }
}

double u(double x, double y) {
    return std::sin(M_PI * x) * std::sin(M_PI * y);
}

double f(double x, double y) {
    return -2 * M_PI * M_PI * std::sin(M_PI * x) * std::sin(M_PI * y);
}

double l2_norm(const std::vector<double>& u_ap, const std::vector<double>& u_ex) {
    double E = 0.0;
    for (size_t i = 0; i < u_ap.size(); ++i) {
        E += std::pow(u_ap[i] - u_ex[i], 2);
    }
    return std::sqrt(E);
}

double L2_norm(const std::vector<double>& u_ap, const std::vector<double>& u_ex, double hx, double hy) {
    double E = l2_norm(u_ap, u_ex) * std::sqrt(hx * hy);
    return E;
}

int main(int argc, char* argv[]) {
    
    const int Nx = atoi(argv[1]);

    const double h = 1.0 / (Nx - 1);

    std::vector<double> x(Nx);
    std::vector<double> y(Nx);

    for (int i = 0; i < Nx; ++i) {
        x[i] = i * h;
        y[i] = i * h;
    }

    const int m = Nx;
    std::vector<std::vector<double>> A(m * m, std::vector<double>(m * m, 0.0));
    std::vector<double> F(m * m, 0.0);

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < m; ++j) {
            int k = i * m + j;
            if (i == 0 || i == m - 1 || j == 0 || j == m - 1) {
                A[k][k] = 1.0;
                F[k] = 0.0;
            } else {
                A[k][k] = -4.0 / (h * h);
                A[k][k - 1] = 1.0 / (h * h);
                A[k][k + 1] = 1.0 / (h * h);
                A[k][k - m] = 1.0 / (h * h);
                A[k][k + m] = 1.0 / (h * h);
                F[k] = f(x[i], y[j]);
            }
        }
    }

    //std::vector<double> u_ap(m * m);
    //
    
    // Compute the analytical solution on the grid
    std::vector<double> xAnalytical(Nx * Nx, 0.0);
    double ha = 1.0 / (Nx - 1);
    for (int i = 0; i < Nx; ++i) {
        for (int j = 0; j < Nx; ++j) {
            double x = i * ha;
            double y = j * ha;
            xAnalytical[i * Nx + j] = u(x, y);
        }
    }

    // Initialize the solution vector
    std::vector<double> sol(Nx * Nx, 0.0);


	
    //Log Time
    //double itime, ftime, exec_time;
    //itime = omp_get_wtime();
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    // Solve using Conjugate Gradient
    bool success = conjugateGradient(A, F, sol, 1e-6, 2*Nx);
    //bool success = parallelConjugateGradient(A, b, x, 1e-6, 10000);


    std::cout << "======================= For Grid Size = "<< Nx  <<" =======================" << std::endl;

    //End Clock
    //ftime = omp_get_wtime();
    //exec_time = ftime - itime;
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "(ms)" << std::endl;

    // Output the solution vector and other computations
    if (success) {

        //std::cout << "Solution found: " << std::endl;
	//std::cout << "Time Taken(s): " << exec_time << std::endl;

	//Calculate the Error
	// Calculate the relative error
	//double error = relativeError(x, xAnalytical);
	double l2_error = L2_norm(sol, xAnalytical, h,h);

    	// Output the relative error
	//std::cout << "Relative error: " << error << std::endl;
	std::cout << "L2 Error: " << l2_error << std::endl;

	//Print the Solution
        //for (int i = 0; i < N * N; ++i) {
        //std::cout << "x[" << i << "] = " << x[i] << std::endl;

	writeSolutionToFile(sol, Nx);

        }

    else {

        std::cout << "Conjugate Gradient failed to converge within maximum iterations." << std::endl;
    }
   

    return 0;
}


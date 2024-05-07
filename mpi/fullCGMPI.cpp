#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <mpi.h>

using namespace std;

const double NEARZERO = 1.0e-8; // interpretation of "zero"

// Prototypes
std::vector<double> matrixTimesVector(std::vector<std::vector<double>> &A, std::vector<double> &V);
std::vector<double> vectorCombination(double a, std::vector<double> &U, double b, std::vector<double> &V);
double innerProduct(std::vector<double> &U, std::vector<double> &V);
double vectorNorm(std::vector<double> &V);
std::vector<double> conjugateGradientSolver(std::vector<std::vector<double>> &A, std::vector<double> &B);

//======================================================================
// Function to write the solution vector to a file
void writeSolutionToFile(const std::vector<double> &x, int N)
{
    std::ofstream outputFile("solution.txt");
    if (outputFile.is_open())
    {
        for (int i = 0; i < N; ++i)
        {
            for (int j = 0; j < N; ++j)
            {
                outputFile << x[i * N + j] << " ";
            }
            outputFile << std::endl;
        }
        outputFile.close();
    }
    else
    {
        std::cerr << "Error: Unable to open file for writing." << std::endl;
    }
}

double u(double x, double y)
{
    return std::sin(M_PI * x) * std::sin(M_PI * y);
}

double f(double x, double y)
{
    return -2 * M_PI * M_PI * std::sin(M_PI * x) * std::sin(M_PI * y);
}

double l2_norm(const std::vector<double> &u_ap, const std::vector<double> &u_ex)
{
    double E = 0.0;
    for (size_t i = 0; i < u_ap.size(); ++i)
    {
        E += std::pow(u_ap[i] - u_ex[i], 2);
    }
    return std::sqrt(E);
}

double L2_norm(const std::vector<double> &u_ap, const std::vector<double> &u_ex, double hx, double hy)
{
    double E = l2_norm(u_ap, u_ex) * std::sqrt(hx * hy);
    return E;
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    const int Nx = atoi(argv[1]);

    const double h = 1.0 / (Nx - 1);

    std::vector<double> x(Nx);
    std::vector<double> y(Nx);

    for (int i = 0; i < Nx; ++i)
    {
        x[i] = i * h;
        y[i] = i * h;
    }

    const int m = Nx;
    std::vector<std::vector<double>> A(m * m, std::vector<double>(m * m, 0.0));
    std::vector<double> F(m * m, 0.0);

    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < m; ++j)
        {
            int k = i * m + j;
            if (i == 0 || i == m - 1 || j == 0 || j == m - 1)
            {
                A[k][k] = 1.0;
                F[k] = 0.0;
            }
            else
            {
                A[k][k] = -4.0 / (h * h);
                A[k][k - 1] = 1.0 / (h * h);
                A[k][k + 1] = 1.0 / (h * h);
                A[k][k - m] = 1.0 / (h * h);
                A[k][k + m] = 1.0 / (h * h);
                F[k] = f(x[i], y[j]);
            }
        }
    }

    // std::vector<double> u_ap(m * m);
    //

    // Compute the analytical solution on the grid
    std::vector<double> xAnalytical(Nx * Nx, 0.0);
    double ha = 1.0 / (Nx - 1);
    for (int i = 0; i < Nx; ++i)
    {
        for (int j = 0; j < Nx; ++j)
        {
            double x = i * ha;
            double y = j * ha;
            xAnalytical[i * Nx + j] = u(x, y);
        }
    }

    // Initialize the solution vector
    std::vector<double> sol(Nx * Nx, 0.0);

    // Start recording time
    double startTime = MPI_Wtime();

    sol = conjugateGradientSolver(A, F);

    // Stop recording time
    double endTime = MPI_Wtime();
    double elapsedTime = endTime - startTime;

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0)
    {
	std::cout << "================== Grid Size = " << Nx << "====================" << std::endl;
        std::cout << "Time taken: " << elapsedTime << " seconds" << std::endl;
        double l2_error = L2_norm(sol, xAnalytical, h, h);
        std::cout << "L2 Error: " << l2_error << std::endl;
        writeSolutionToFile(sol, Nx);
    }

    // double l2_error = L2_norm(sol, xAnalytical, h, h);
    // std::cout << "L2 Error: " << l2_error << std::endl;
    // writeSolutionToFile(sol, Nx);

    MPI_Finalize();

    return 0;
}

//======================================================================

std::vector<double> matrixTimesVector(std::vector<std::vector<double>> &A, std::vector<double> &V)
{
    int n = A.size();
    std::vector<double> C(n, 0.0);
    for (int i = 0; i < n; i++)
        C[i] = innerProduct(A[i], V);
    return C;
}

//======================================================================

std::vector<double> vectorCombination(double a, std::vector<double> &U, double b, std::vector<double> &V)
{
    int n = U.size();
    std::vector<double> W(n, 0.0);
    for (int j = 0; j < n; j++)
        W[j] = a * U[j] + b * V[j];
    return W;
}

//======================================================================

double innerProduct(std::vector<double> &U, std::vector<double> &V) // Inner product of U and V
{
    int mpi_size, mpi_rank;

    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    int size = U.size();

    int local_size = size / mpi_size;
    int extra = size % mpi_size;

    // Distribute data among processes
    std::vector<double> local_U(local_size);
    std::vector<double> local_V(local_size);
    MPI_Scatter(U.data(), local_size, MPI_DOUBLE, local_U.data(), local_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(V.data(), local_size, MPI_DOUBLE, local_V.data(), local_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Compute local inner product
    double local_sum = 0.0;
    for (int i = 0; i < local_size; i++)
    {
        local_sum += local_U[i] * local_V[i];
    }

    // Sum up local sums to get the global sum
    double global_sum;
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    return global_sum;
}

//======================================================================

double vectorNorm(std::vector<double> &V) // Vector norm
{
    return sqrt(innerProduct(V, V));
}

//======================================================================

std::vector<double> conjugateGradientSolver(std::vector<std::vector<double>> &A, std::vector<double> &B)
{
    double TOLERANCE = 1.0e-6;

    int n = A.size();
    std::vector<double> X(n, 0.0);
    std::vector<double> AP(n, 0.0);
    std::vector<double> R = B;
    std::vector<double> P = R;
    double alpha, beta, rsold, rsnew;
    int k = 0;

    while (k < n)
    {
        std::vector<double> Rold = R; // Store previous residual
        std::vector<double> AP = matrixTimesVector(A, P);
        double alpha = innerProduct(R, R) / innerProduct(P, AP);

        X = vectorCombination(1.0, X, alpha, P);   // Next estimate of solution
        R = vectorCombination(1.0, R, -alpha, AP); // Residual

        if (vectorNorm(R) < TOLERANCE)
        {
            break;
        }

        double beta = innerProduct(R, R) / innerProduct(Rold, Rold);

        P = vectorCombination(1.0, R, beta, P); // Next gradient
        k++;
    }

    return X;
}

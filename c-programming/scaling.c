#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include <lapacke.h>
#include <cblas.h>

/*
   1) Compute the Frobenius norm of a matrix (used for error computation)
*/
static double frobenius_norm(const double *A, int m, int n)
{
    double sum = 0.0;
    for (int i = 0; i < m * n; i++)
        sum += A[i] * A[i];
    return sqrt(sum);
}

/*
   2) Core TSQR function (simplified version)
   - Assumes the rows are evenly distributed among processes
   - As required in Q2: Perform local QR and then gather the R's for a global QR
*/
static void TSQR_MPI(double *A_local, int M_local, int N, MPI_Comm comm, double *R_final)
{
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // (A) Local QR factorization
    double *tau_local = (double *)malloc(N * sizeof(double));
    LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, M_local, N, A_local, N, tau_local);

    // Extract the local R (only the upper triangular part)
    double *R_local = (double *)calloc(N * N, sizeof(double));
    for (int i = 0; i < N; i++)
        for (int j = i; j < N; j++)
            R_local[i * N + j] = A_local[i * N + j];

    // Generate the local Q (A_local is overwritten to become Q_local)
    LAPACKE_dorgqr(LAPACK_ROW_MAJOR, M_local, N, N, A_local, N, tau_local);
    free(tau_local);

    // --- (B) Gather all processes' local R into the root process and perform a QR on them to obtain the global R ---
    double *R_stack = NULL;
    if (rank == 0)
        R_stack = (double *)malloc(size * N * N * sizeof(double));

    MPI_Gather(R_local, N * N, MPI_DOUBLE, R_stack, N * N, MPI_DOUBLE, 0, comm);
    free(R_local);

    double *Q_global = NULL;
    if (rank == 0)
    {
        int rstack_rows = size * N;
        double *tau_global = (double *)malloc(N * sizeof(double));

        // Perform QR factorization on R_stack
        LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, rstack_rows, N, R_stack, N, tau_global);

        // Extract the final R
        memset(R_final, 0, N * N * sizeof(double));
        for (int i = 0; i < N; i++)
            for (int j = i; j < N; j++)
                R_final[i * N + j] = R_stack[i * N + j];

        // Generate the global Q
        Q_global = (double *)malloc(rstack_rows * N * sizeof(double));
        memcpy(Q_global, R_stack, rstack_rows * N * sizeof(double));
        LAPACKE_dorgqr(LAPACK_ROW_MAJOR, rstack_rows, N, N, Q_global, N, tau_global);

        free(tau_global);
        free(R_stack);
    }

    // (C) Scatter the corresponding block of global Q to each process and multiply it with the local Q
    double *Q_global_local = (double *)malloc(N * N * sizeof(double));
    MPI_Scatter(Q_global, N * N, MPI_DOUBLE, Q_global_local, N * N, MPI_DOUBLE, 0, comm);
    if (rank == 0)
        free(Q_global);

    // Local Q_final = local Q * Q_global_local
    double *Q_final_local = (double *)malloc(M_local * N * sizeof(double));
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M_local, N, N,
                1.0, A_local, N, Q_global_local, N,
                0.0, Q_final_local, N);

    free(Q_global_local);
    memcpy(A_local, Q_final_local, M_local * N * sizeof(double));
    free(Q_final_local);

    // (D) Broadcast the final R to all processes
    MPI_Bcast(R_final, N * N, MPI_DOUBLE, 0, comm);
}

/*
   3) Main function: Loop over multiple (M, N) test cases and write results to scaling_results.txt
*/
int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // 1) Predefine a set of (M, N) test values; modify as needed
    int M_values[] = {1000, 10000, 100000, 1000000, 10000000};
    int N_values[] = {4, 8, 16, 32};
    int num_M = sizeof(M_values) / sizeof(M_values[0]);
    int num_N = sizeof(N_values) / sizeof(N_values[0]);

    // 2) Root process opens the file scaling_results.txt for writing
    FILE *fp = NULL;
    if (rank == 0)
    {
        fp = fopen("scaling_results.txt", "w");
        if (!fp)
        {
            fprintf(stderr, "Error: Cannot open scaling_results.txt for writing.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        fprintf(fp, "TSQR Scaling Test\n");
        fprintf(fp, "Number of processes = %d\n", size);
        fprintf(fp, "-----------------------------------------------------\n");
        fprintf(fp, "  M      N    Time(s)    AbsError       RelError\n");
        fprintf(fp, "-----------------------------------------------------\n");
    }

    // 3) Begin looping over the test cases
    for (int i = 0; i < num_M; i++)
    {
        for (int j = 0; j < num_N; j++)
        {
            int M_global = M_values[i];
            int N = N_values[j];

            // Assume M_global is divisible by size (as specified: rows are evenly distributed)
            if (M_global % size != 0)
            {
                if (rank == 0)
                    fprintf(fp, "Warning: M_global=%d not divisible by size=%d!\n", M_global, size);
                // Simple handling: skip this test case
                continue;
            }
            int M_local = M_global / size;

            // Generate A_global and make a copy of it
            double *A_global = NULL;
            double *A_global_copy = NULL;
            if (rank == 0)
            {
                A_global = (double *)malloc(M_global * N * sizeof(double));
                A_global_copy = (double *)malloc(M_global * N * sizeof(double));
                srand(12345);
                for (int k = 0; k < M_global * N; k++)
                {
                    double val = (double)rand() / RAND_MAX;
                    A_global[k] = val;
                    A_global_copy[k] = val;
                }
            }

            // Scatter A_global to all processes
            double *A_local = (double *)malloc(M_local * N * sizeof(double));
            MPI_Scatter(A_global, M_local * N, MPI_DOUBLE,
                        A_local, M_local * N, MPI_DOUBLE,
                        0, MPI_COMM_WORLD);

            // Allocate memory for R
            double *R_final = (double *)malloc(N * N * sizeof(double));

            // Start timing
            MPI_Barrier(MPI_COMM_WORLD);
            double t_start = MPI_Wtime();

            // Execute TSQR
            TSQR_MPI(A_local, M_local, N, MPI_COMM_WORLD, R_final);

            // End timing
            MPI_Barrier(MPI_COMM_WORLD);
            double t_end = MPI_Wtime();
            double local_time = t_end - t_start;

            // Gather Q from all processes
            double *Q_global = NULL;
            if (rank == 0)
                Q_global = (double *)malloc(M_global * N * sizeof(double));

            MPI_Gather(A_local, M_local * N, MPI_DOUBLE,
                       Q_global, M_local * N, MPI_DOUBLE,
                       0, MPI_COMM_WORLD);

            // Use the maximum time among processes (the slowest determines the overall time)
            double total_time;
            MPI_Reduce(&local_time, &total_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

            // Root process computes the error and writes results to file
            if (rank == 0)
            {
                // Reconstruct A = Q * R
                double *A_reconstructed = (double *)malloc(M_global * N * sizeof(double));
                cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                            M_global, N, N,
                            1.0, Q_global, N,
                            R_final, N,
                            0.0, A_reconstructed, N);

                double error = 0.0;
                for (int k = 0; k < M_global * N; k++)
                {
                    double diff = A_global_copy[k] - A_reconstructed[k];
                    error += diff * diff;
                }
                error = sqrt(error);
                double normA = frobenius_norm(A_global_copy, M_global, N);
                double rel_error = error / normA;

                // Write the results to file
                fprintf(fp, "%6d %6d %10.6f  %10.3e  %10.3e\n",
                        M_global, N, total_time, error, rel_error);

                free(A_reconstructed);
                free(A_global);
                free(A_global_copy);
                free(Q_global);
            }

            free(A_local);
            free(R_final);
        }
    }

    // 4) Close the file
    if (rank == 0)
    {
        fprintf(fp, "-----------------------------------------------------\n");
        fclose(fp);
        printf("Scaling results written to scaling_results.txt\n");
    }

    MPI_Finalize();
    return 0;
}

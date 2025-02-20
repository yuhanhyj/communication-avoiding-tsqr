#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include <lapacke.h>
#include <cblas.h>

// Helper function to print the first few rows of a matrix
static void print_matrix_partial(const char *name, const double *A, int rows, int cols, int max_rows)
{
    printf("\n%s (showing up to %d rows):\n", name, max_rows);
    int print_rows = (rows < max_rows ? rows : max_rows);
    for (int i = 0; i < print_rows; i++)
    {
        for (int j = 0; j < cols; j++)
            printf("%10.4f ", A[i * cols + j]);
        printf("\n");
    }
    if (rows > max_rows)
        printf("  ... (total %d rows, truncated)\n", rows);
}

// Compute the Frobenius norm of a matrix
double frobenius_norm(const double *A, int m, int n)
{
    double sum = 0.0;
    for (int i = 0; i < m * n; i++)
        sum += A[i] * A[i];
    return sqrt(sum);
}

// Core TSQR function
void TSQR_MPI(double *A_local, int M_local, int N, MPI_Comm comm, double *R_final)
{
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // 1) Local QR factorization
    double *tau_local = (double *)malloc(N * sizeof(double));
    LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, M_local, N, A_local, N, tau_local);

    // Extract local R (only the upper triangular part)
    double *R_local = (double *)calloc(N * N, sizeof(double));
    for (int i = 0; i < N; i++)
        for (int j = i; j < N; j++)
            R_local[i * N + j] = A_local[i * N + j];

    // Generate local Q (A_local is overwritten to become Q_local)
    LAPACKE_dorgqr(LAPACK_ROW_MAJOR, M_local, N, N, A_local, N, tau_local);
    free(tau_local);

    // 2) Gather all processes' local R into the root process and perform a QR on them
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

        // Extract final R
        memset(R_final, 0, N * N * sizeof(double));
        for (int i = 0; i < N; i++)
            for (int j = i; j < N; j++)
                R_final[i * N + j] = R_stack[i * N + j];

        // Generate global Q
        Q_global = (double *)malloc(rstack_rows * N * sizeof(double));
        memcpy(Q_global, R_stack, rstack_rows * N * sizeof(double));
        LAPACKE_dorgqr(LAPACK_ROW_MAJOR, rstack_rows, N, N, Q_global, N, tau_global);

        free(tau_global);
        free(R_stack);
    }

    // 3) Scatter the corresponding block of global Q to each process
    double *Q_global_local = (double *)malloc(N * N * sizeof(double));
    MPI_Scatter(Q_global, N * N, MPI_DOUBLE, Q_global_local, N * N, MPI_DOUBLE, 0, comm);
    if (rank == 0)
        free(Q_global);

    // 4) Final Q_local = Q_local * Q_global_local
    double *Q_final_local = (double *)malloc(M_local * N * sizeof(double));
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M_local, N, N,
                1.0, A_local, N, Q_global_local, N,
                0.0, Q_final_local, N);

    free(Q_global_local);
    memcpy(A_local, Q_final_local, M_local * N * sizeof(double));
    free(Q_final_local);

    // 5) Broadcast the final R to all processes
    MPI_Bcast(R_final, N * N, MPI_DOUBLE, 0, comm);
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Default parameters
    int M_global = 16, N = 4;
    if (argc >= 3)
    {
        M_global = atoi(argv[1]);
        N = atoi(argv[2]);
    }

    if (rank == 0)
        printf("Global matrix: M = %d, N = %d\n", M_global, N);

    // Calculate the number of rows assigned to each process
    int base = M_global / size;
    int rem = M_global % size;
    int M_local = base + ((rank < rem) ? 1 : 0);

    // Only the root process generates A_global and A_global_copy
    double *A_global = NULL, *A_global_copy = NULL;
    int *sendcounts = NULL;
    int *displs = NULL;
    if (rank == 0)
    {
        A_global = (double *)malloc(M_global * N * sizeof(double));
        A_global_copy = (double *)malloc(M_global * N * sizeof(double));

        // Initialize the original A matrix
        srand(12345);
        for (int i = 0; i < M_global * N; i++)
        {
            double val = (double)rand() / RAND_MAX;
            A_global[i] = val;
            A_global_copy[i] = val;
        }

        // Print the first 8 rows of the original A (optional)
        print_matrix_partial("Original A", A_global, M_global, N, 8);

        // Prepare parameters for MPI_Scatterv / MPI_Gatherv
        sendcounts = (int *)malloc(size * sizeof(int));
        displs = (int *)malloc(size * sizeof(int));

        int offset = 0;
        for (int i = 0; i < size; i++)
        {
            int rows_i = base + ((i < rem) ? 1 : 0);
            sendcounts[i] = rows_i * N;
            displs[i] = offset;
            offset += sendcounts[i];
        }
    }

    // Each process allocates local A_local
    double *A_local = (double *)malloc(M_local * N * sizeof(double));

    // Distribute A_global to all processes
    MPI_Scatterv(A_global, sendcounts, displs, MPI_DOUBLE,
                 A_local, M_local * N, MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    // Allocate space for the final R
    double *R_final = (double *)malloc(N * N * sizeof(double));

    // TSQR computation
    double t_start = MPI_Wtime();
    TSQR_MPI(A_local, M_local, N, MPI_COMM_WORLD, R_final);
    double t_end = MPI_Wtime();

    // The root process performs error analysis
    if (rank == 0)
    {
        printf("TSQR MPI Time: %f seconds\n", t_end - t_start);

        // Print the final R (4x4)
        printf("\nFinal R (size %dx%d):\n", N, N);
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
                printf("%10.4f ", R_final[i * N + j]);
            printf("\n");
        }

        // Gather the Q factors from all processes
        double *Q_global_final = (double *)malloc(M_global * N * sizeof(double));
        MPI_Gatherv(A_local, M_local * N, MPI_DOUBLE,
                    Q_global_final, sendcounts, displs, MPI_DOUBLE,
                    0, MPI_COMM_WORLD);

        // Compute A_reconstructed = Q * R
        double *A_reconstructed = (double *)malloc(M_global * N * sizeof(double));
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    M_global, N, N,
                    1.0, Q_global_final, N,
                    R_final, N,
                    0.0, A_reconstructed, N);

        // Print the first 8 rows of the reconstructed A (optional)
        print_matrix_partial("Reconstructed A", A_reconstructed, M_global, N, 8);

        // Compute the error
        double error = 0.0;
        for (int i = 0; i < M_global * N; i++)
        {
            double diff = A_global_copy[i] - A_reconstructed[i];
            error += diff * diff;
        }
        error = sqrt(error);
        double normA = frobenius_norm(A_global_copy, M_global, N);
        double rel_error = error / normA;

        // Print the errors
        printf("\nFinal Error (absolute): %e\n", error);
        printf("Final Error (relative): %e\n", rel_error);

        // Free root process data
        free(Q_global_final);
        free(A_reconstructed);
        free(A_global);
        free(A_global_copy);
        free(sendcounts);
        free(displs);
    }
    else
    {
        // Non-root processes execute MPI_Gatherv
        MPI_Gatherv(A_local, M_local * N, MPI_DOUBLE,
                    NULL, NULL, NULL, MPI_DOUBLE,
                    0, MPI_COMM_WORLD);
    }

    free(A_local);
    free(R_final);

    MPI_Finalize();
    return 0;
}

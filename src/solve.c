#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include "custom.h"

typedef struct {
    double total_time;
    int calls;
} TimeStats;

TimeStats spmv_stats = {0};
TimeStats dot_stats = {0};
TimeStats axpy_stats = {0};

void copy_vec(int N, double* x, double* y) {
    for (int i = 0; i < N; ++i) {
        y[i] = x[i];
    }
}

void fill_constant(int N, double alpha, double* x) {
    for (int i = 0; i < N; ++i) {
        x[i] = alpha;
    }
}

void spmv(
    int N, 
    int* IA, 
    int* JA, 
    double* A, 
    double* x, 
    double* y, 
    int T, 
    FILE* out
) {
    double start_time = omp_get_wtime();
    omp_set_num_threads(T);

    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        double sum = 0.0;
        for (int k = IA[i]; k < IA[i + 1]; ++k) {
            sum += A[k] * x[JA[k]];
        }
        y[i] = sum;
    }

    double elapsed = omp_get_wtime() - start_time;
    spmv_stats.total_time += elapsed;
    spmv_stats.calls++;
}

double dot(
    int N, 
    double* x, 
    double* y, 
    int T, 
    FILE* out
) {
    double start_time = omp_get_wtime();
    omp_set_num_threads(T);

    double result = 0.0;

    #pragma omp parallel for reduction(+:result)
    for (int i = 0; i < N; ++i) {
        result += x[i] * y[i];
    }

    double elapsed = omp_get_wtime() - start_time;
    dot_stats.total_time += elapsed;
    dot_stats.calls++;

    return result;
}

void axpy(
    int N, 
    double alpha, 
    double* x, 
    double* y, 
    int T, 
    FILE* out
) {
    double start_time = omp_get_wtime();
    omp_set_num_threads(T);

    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        y[i] += alpha * x[i];
    }

    double elapsed = omp_get_wtime() - start_time;
    axpy_stats.total_time += elapsed;
    axpy_stats.calls++;
}

void Solve(
    double* A,
    double* b,
    double* x,
    int N,
    int* IA,
    int* JA,
    double eps,
    int maxit,
    int* n,
    double* res,
    int T,
    FILE* out
) {
    omp_set_num_threads(T);

    double start_total = omp_get_wtime();

    double* r = (double*)malloc(N * sizeof(double));
    double* z = (double*)malloc(N * sizeof(double));
    double* p = (double*)malloc(N * sizeof(double));
    double* q = (double*)malloc(N * sizeof(double));
    double* M = (double*)malloc(N * sizeof(double));

    if (!r || !z || !p || !q || !M) {
        fprintf(out, "failed to allocate memory in Solve\n");
        return;
    }

    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        for (int k = IA[i]; k < IA[i + 1]; ++k) {
            if (JA[k] == i) {
                M[i] = 1.0 / A[k];
                break;
            }
        }
    }

    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        x[i] = 0.0;
        r[i] = b[i];
    }

    double rho = 0.0, rho_prev = 0.0, alpha = 0.0, beta = 0.0;
    int k = 0;

    do {
        ++k;

        #pragma omp parallel for
        for (int i = 0; i < N; ++i) {
            z[i] = M[i] * r[i];
        }

        rho = dot(N, r, z, T, out);

        if (k == 1) {
            #pragma omp parallel for
            for (int i = 0; i < N; ++i) {
                p[i] = z[i];
            }
        } else {
            beta = rho / rho_prev;
            axpy(N, beta, p, z, T, out);
            #pragma omp parallel for
            for (int i = 0; i < N; ++i) {
                p[i] = z[i];
            }
        }

        spmv(N, IA, JA, A, p, q, T, out);
        alpha = rho / dot(N, p, q, T, out);
        axpy(N, alpha, p, x, T, out);
        axpy(N, -alpha, q, r, T, out);

        rho_prev = rho;

    } while (rho > eps * eps && k < maxit);

    spmv(N, IA, JA, A, x, q, T, out);
    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        r[i] = q[i] - b[i];
    }
    *res = sqrt(dot(N, r, r, T, out));
    *n = k;

    double end_total = omp_get_wtime();
    fprintf(out, "Solve: %e seconds\n", end_total - start_total);

    fprintf(out, "\nSPMV: %e seconds\n", spmv_stats.total_time);
    fprintf(out, "  Total calls: %d\n", spmv_stats.calls);
    fprintf(out, "  Average time: %e seconds\n", spmv_stats.total_time / spmv_stats.calls);

    fprintf(out, "\nDOT: %e\n", dot_stats.total_time);
    fprintf(out, "  Total calls: %d\n", dot_stats.calls);
    fprintf(out, "  Average time: %e seconds\n", dot_stats.total_time / dot_stats.calls);

    fprintf(out, "\nAXPY: %e\n", axpy_stats.total_time);
    fprintf(out, "  Total calls: %d\n", axpy_stats.calls);
    fprintf(out, "  Average time: %e seconds\n", axpy_stats.total_time / axpy_stats.calls);

    free(r);
    free(z);
    free(p);
    free(q);
    free(M);
}
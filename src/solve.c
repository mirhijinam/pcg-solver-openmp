#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

typedef struct {
    double total_time;
    int calls;
} TimeStats;

static TimeStats spmv_stats = {0};
static TimeStats dot_stats = {0};
static TimeStats axpy_stats = {0};

static inline void copy_vec(int N, const double *restrict x, double *restrict y) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i) {
        y[i] = x[i];
    }
}

static inline void fill_constant(int N, double alpha, double *restrict x) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i) {
        x[i] = alpha;
    }
}

static void spmv(
    int N, 
    const int *restrict IA, 
    const int *restrict JA, 
    const double *restrict A, 
    const double *restrict x, 
    double *restrict y, 
    FILE* out
) {
    double start_time = omp_get_wtime();

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i) {
        double sum = 0.0;
        for (int k = IA[i]; k < IA[i + 1]; ++k) {
            sum += A[k] * x[JA[k]];
        }
        y[i] = sum;
    }

    double elapsed = omp_get_wtime() - start_time;
    #pragma omp atomic
    spmv_stats.total_time += elapsed;
    #pragma omp atomic
    spmv_stats.calls++;
}

static double dot(
    int N, 
    const double *restrict x, 
    const double *restrict y, 
    FILE* out
) {
    double start_time = omp_get_wtime();

    double result = 0.0;

    #pragma omp parallel for schedule(static) reduction(+:result)
    for (int i = 0; i < N; ++i) {
        result += x[i] * y[i];
    }

    double elapsed = omp_get_wtime() - start_time;
    #pragma omp atomic
    dot_stats.total_time += elapsed;
    #pragma omp atomic
    dot_stats.calls++;

    return result;
}

static void axpy(
    int N, 
    double alpha, 
    const double *restrict x, 
    double *restrict y, 
    FILE* out
) {
    double start_time = omp_get_wtime();

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i) {
        y[i] += alpha * x[i];
    }

    double elapsed = omp_get_wtime() - start_time;
    #pragma omp atomic
    axpy_stats.total_time += elapsed;
    #pragma omp atomic
    axpy_stats.calls++;
}

void Solve(
    double *restrict A,
    double *restrict b,
    double *restrict x,
    int N,
    int *restrict IA,
    int *restrict JA,
    double eps,
    int maxit,
    int *n,
    double *res,
    int T,
    FILE *out
) {
    // Предполагается, что omp_set_num_threads(T) вызван до вызова Solve или в main

    double start_total = omp_get_wtime();

    double *restrict r = (double*)malloc(N * sizeof(double));
    double *restrict z = (double*)malloc(N * sizeof(double));
    double *restrict p = (double*)malloc(N * sizeof(double));
    double *restrict q = (double*)malloc(N * sizeof(double));
    double *restrict M = (double*)malloc(N * sizeof(double));

    if (!r || !z || !p || !q || !M) {
        free(r);
        free(z);
        free(p);
        free(q);
        free(M);
        return;
    }

    // Одним параллельным регионом вычисляем M, x, r
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i) {
        // Поиск диагонального элемента
        double diag = 0.0;
        for (int k = IA[i]; k < IA[i + 1]; ++k) {
            if (JA[k] == i) {
                diag = A[k];
                break;
            }
        }
        M[i] = 1.0 / diag;
        x[i] = 0.0;
        r[i] = b[i];
    }

    double rho = 0.0, rho_prev = 0.0, alpha = 0.0, beta = 0.0;
    int iter = 0;

    while (1) {
        iter++;

        // z = M * r
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < N; ++i) {
            z[i] = M[i] * r[i];
        }

        rho = dot(N, r, z, out);

        if (iter == 1) {
            // p = z
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < N; ++i) {
                p[i] = z[i];
            }
        } else {
            beta = rho / rho_prev;
            // p = z + beta * p
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < N; ++i) {
                p[i] = z[i] + beta * p[i];
            }
        }

        spmv(N, IA, JA, A, p, q, out);
        double pq = dot(N, p, q, out);
        alpha = rho / pq;

        axpy(N, alpha, p, x, out);
        axpy(N, -alpha, q, r, out);

        // Проверка на сходимость
        if (rho < eps * eps || iter >= maxit)
            break;

        rho_prev = rho;
    }

    // Вычисляем невязку
    spmv(N, IA, JA, A, x, q, out);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i) {
        r[i] = q[i] - b[i];
    }
    *res = sqrt(dot(N, r, r, out));
    *n = iter;

    double end_total = omp_get_wtime();
    fprintf(out, "Solve: %e seconds\n", end_total - start_total);

    fprintf(out, "\nSPMV: %e seconds\n", spmv_stats.total_time);
    fprintf(out, "  Total calls: %d\n", spmv_stats.calls);
    fprintf(out, "  Average time: %e seconds\n", spmv_stats.total_time / (spmv_stats.calls ? spmv_stats.calls : 1));

    fprintf(out, "DOT: %e seconds\n", dot_stats.total_time);
    fprintf(out, "  Total calls: %d\n", dot_stats.calls);
    fprintf(out, "  Average time: %e seconds\n", dot_stats.total_time / (dot_stats.calls ? dot_stats.calls : 1));

    fprintf(out, "AXPY: %e seconds\n", axpy_stats.total_time);
    fprintf(out, "  Total calls: %d\n", axpy_stats.calls);
    fprintf(out, "  Average time: %e seconds\n", axpy_stats.total_time / (axpy_stats.calls ? axpy_stats.calls : 1));

    free(r);
    free(z);
    free(p);
    free(q);
    free(M);
}

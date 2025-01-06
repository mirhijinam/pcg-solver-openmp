#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

void axpy(int N, double alpha, double* x, double* y, int T, FILE* out) {
    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        y[i] += alpha * x[i];
    }
}

double dot(int N, double* x, double* y, int T, FILE* out) {
    double result = 0.0;
    #pragma omp parallel for reduction(+:result)
    for (int i = 0; i < N; ++i) {
        result += x[i] * y[i];
    }
    return result;
}

void spmv(int N, int* IA, int* JA, double* A,
          double* x, double* y, int T, FILE* out) {
    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        double sum = 0.0;
        for (int k = IA[i]; k < IA[i + 1]; ++k) {
            sum += A[k] * x[JA[k]];
        }
        y[i] = sum;
    }
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
    double start_total, end_total;
    double time_spmv = 0.0, time_axpy = 0.0, time_dot = 0.0;

    start_total = omp_get_wtime();

    double* r = (double*)malloc(N * sizeof(double));
    double* z = (double*)malloc(N * sizeof(double));
    double* p = (double*)malloc(N * sizeof(double));
    double* q = (double*)malloc(N * sizeof(double));
    double* M = (double*)malloc(N * sizeof(double));

    if (!r || !z || !p || !q || !M) {
        fprintf(out, "failed to allocate memory in Solve\n");
        free(r); free(z); free(p); free(q); free(M);
        return;
    }

    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        M[i] = 1.0;
        for (int k = IA[i]; k < IA[i + 1]; ++k) {
            if (JA[k] == i) {
                M[i] = 1.0 / A[k];
                break;
            }
        }
    }

    // x_0 = 0, r_0 = b
    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        x[i] = 0.0;
        r[i] = b[i];
    }

    double rho = 0.0, rho_prev = 0.0, alpha = 0.0, beta = 0.0;
    int k = 0;

    do {
        ++k;

        // z = M^(-1)*r
        #pragma omp parallel for
        for (int i = 0; i < N; ++i) {
            z[i] = M[i] * r[i];
        }

        // rho = dot(r,z)
        double start_phase = omp_get_wtime();
        rho = dot(N, r, z, T, out);
        double end_phase = omp_get_wtime();
        time_dot += (end_phase - start_phase);

        if (k == 1) {
            // p = z
            #pragma omp parallel for
            for (int i = 0; i < N; ++i) {
                p[i] = z[i];
            }
        } else {
            beta = rho / rho_prev;

            // p <- z + beta*p
            start_phase = omp_get_wtime();
            axpy(N, beta, p, z, T, out);  // z = z + beta*p
            end_phase = omp_get_wtime();
            time_axpy += (end_phase - start_phase);

            #pragma omp parallel for
            for (int i = 0; i < N; ++i) {
                p[i] = z[i];
            }
        }

        // q = A*p
        start_phase = omp_get_wtime();
        spmv(N, IA, JA, A, p, q, T, out);
        end_phase = omp_get_wtime();
        time_spmv += (end_phase - start_phase);

        // alpha = rho / dot(p,q)
        start_phase = omp_get_wtime();
        alpha = rho / dot(N, p, q, T, out);
        end_phase = omp_get_wtime();
        time_dot += (end_phase - start_phase);

        // x <- x + alpha*p
        start_phase = omp_get_wtime();
        axpy(N, alpha, p, x, T, out);
        end_phase = omp_get_wtime();
        time_axpy += (end_phase - start_phase);

        // r <- r - alpha*q
        start_phase = omp_get_wtime();
        axpy(N, -alpha, q, r, T, out);
        end_phase = omp_get_wtime();
        time_axpy += (end_phase - start_phase);

        rho_prev = rho;

    } while (rho > eps * eps && k < maxit);

    // Финальная невязка: ещё один spmv
    double start_phase = omp_get_wtime();
    spmv(N, IA, JA, A, x, q, T, out);  // q = A*x
    double end_phase = omp_get_wtime();
    time_spmv += (end_phase - start_phase);

    // r = q - b
    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        r[i] = q[i] - b[i];
    }

    // res = ||r||
    start_phase = omp_get_wtime();
    *res = sqrt(dot(N, r, r, T, out));
    end_phase = omp_get_wtime();
    time_dot += (end_phase - start_phase);

    *n = k;

    end_total = omp_get_wtime();

    fprintf(out, "Solve:%f\n", end_total - start_total);
    fprintf(out, "\tSPMV:%f\n", time_spmv);
    fprintf(out, "\tAXPY:%f\n", time_axpy);
    fprintf(out, "\tDOT:%f\n", time_dot);

    free(r);
    free(z);
    free(p);
    free(q);
    free(M);
}

void spmv_seq(
    int N,
    int* IA,
    int* JA,
    double* A,
    double* x,
    double* y,
    FILE* out
) {
    for (int i = 0; i < N; ++i) {
        y[i] = 0.0;
        for (int k = IA[i]; k < IA[i + 1]; ++k) {
            y[i] += A[k] * x[JA[k]];
        }
    }
}

double dot_seq(
    int N,
    double* x,
    double* y,
    FILE* out
) {
    double result = 0.0;
    for (int i = 0; i < N; ++i) {
        result += x[i] * y[i];
    }

    return result;
}

void axpy_seq(
    int N,
    double alpha,
    double* x,
    double* y,
    FILE* out
) {
    for (int i = 0; i < N; ++i) {
        y[i] += alpha * x[i];
    }
}

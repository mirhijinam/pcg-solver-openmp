#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include "custom.h"

void copy_vec(
    int N,
    double* x,
    double* y
) {
    for (int i = 0; i < N; ++i) {
        y[i] = x[i];
    }
}

void fill_constant(
    int N,
    double alpha,
    double* x
) {
    for (int i = 0; i < N; ++i) {
        x[i] = alpha;
    }
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
#ifdef DBG_SOLVER
    double start_time = omp_get_wtime();
#endif
    
    for (int i = 0; i < N; ++i) {
        y[i] = 0.0;
        for (int k = IA[i]; k < IA[i + 1]; ++k) {
            y[i] += A[k] * x[JA[k]];
        }
    }
    
#ifdef DBG_SOLVER
    double end_time = omp_get_wtime();
    fprintf(out, "spmv_seq time: %f seconds\n", end_time - start_time);
#endif
}

double dot_seq(
    int N,
    double* x,
    double* y,
    FILE* out
) {
#ifdef DBG_SOLVER
    double start_time = omp_get_wtime();
#endif
    
    double result = 0.0;
    for (int i = 0; i < N; ++i) {
        result += x[i] * y[i];
    }
    
#ifdef DBG_SOLVER
    double end_time = omp_get_wtime();
    fprintf(out, "dot_seq time: %f seconds\n", end_time - start_time);
#endif
    
    return result;
}

void axpy_seq(
    int N,
    double alpha,
    double* x,
    double* y,
    FILE* out
) {
#ifdef DBG_SOLVER
    double start_time = omp_get_wtime();
#endif
    
    for (int i = 0; i < N; ++i) {
        y[i] += alpha * x[i];
    }
    
#ifdef DBG_SOLVER
    double end_time = omp_get_wtime();
    fprintf(out, "axpy_seq time: %f seconds\n", end_time - start_time);
#endif
}

void axpy(
    int N,
    double alpha,
    double* x,
    double* y,
    int T,
    FILE* out
) {
    omp_set_num_threads(T);

#ifdef DBG_SOLVER
    double start_time = omp_get_wtime();
#endif
    
    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        y[i] += alpha * x[i];
    }
    
#ifdef DBG_SOLVER
    double end_time = omp_get_wtime();
    fprintf(out, "axpy time: %f seconds\n", end_time - start_time);
#endif
}

double dot(
    int N,
    double* x,
    double* y,
    int T,
    FILE* out
) {
    omp_set_num_threads(T);

#ifdef DBG_SOLVER
    double start_time = omp_get_wtime();
#endif
    
    double result = 0.0;
    #pragma omp parallel
    {
        double local_sum = 0.0;
        #pragma omp for
        for (int i = 0; i < N; ++i) {
            local_sum += x[i] * y[i];
        }
        #pragma omp atomic
        result += local_sum;
    }

#ifdef DBG_SOLVER
    double end_time = omp_get_wtime();
    fprintf(out, "dot time: %f seconds\n", end_time - start_time);
#endif
    
    return result;
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
    omp_set_num_threads(T);

#ifdef DBG_SOLVER
    double start_time = omp_get_wtime();
#endif
    
    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        double sum = 0.0;
        for (int k = IA[i]; k < IA[i + 1]; ++k) {
            sum += A[k] * x[JA[k]];
        }
        y[i] = sum;
    }
    
#ifdef DBG_SOLVER
    double end_time = omp_get_wtime();
    fprintf(out, "spmv time: %f seconds\n", end_time - start_time);
#endif
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

    double start_total, end_total, start_phase, end_phase;
    start_total = omp_get_wtime();

    double* r = (double*)malloc(N * sizeof(double));
    double* z = (double*)malloc(N * sizeof(double));
    double* p = (double*)malloc(N * sizeof(double));
    double* q = (double*)malloc(N * sizeof(double));
    double* M = (double*)malloc(N * sizeof(double));

    if (!r || !z || !p || !q || !M) {
        fprintf(out, "failed to allocate memory in Solve\n");
        return;
    }

    start_phase = omp_get_wtime();
    // Обратная матрица для диагонального предобуславливателя
    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        for (int k = IA[i]; k < IA[i + 1]; ++k) {
            if (JA[k] == i) {       // диагональный элемент
                M[i] = 1.0 / A[k];  // сразу сохраняем обратное значение
                break;
            }
        }
    }
    end_phase = omp_get_wtime();
    fprintf(out, "Phase 6 (inverse preconditioner) time: %f seconds\n", end_phase - start_phase);

    start_phase = omp_get_wtime();
    // x_0 = 0 и r_0 = b
    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        x[i] = 0.0;
        r[i] = b[i];
    }

    double rho = 0.0, rho_prev = 0.0, alpha = 0.0, beta = 0.0;  // инициализируем все переменные
    int k = 0;

    do {
        ++k;
        
        // z_k = M^(-1) * r_k-1
        #pragma omp parallel for
        for (int i = 0; i < N; ++i) {
            z[i] = M[i] * r[i];         // M - обратная матрица
        }

        // rho_k = (r_k-1,z_k)
        rho = dot(N, r, z, T, out);

        if (k == 1) {
            // p_k = z_k
            #pragma omp parallel for
            for (int i = 0; i < N; ++i) {
                p[i] = z[i];
            }
        } else {
            beta = rho / rho_prev;
            // p_k = z_k + beta_k * p_k-1
            axpy(N, beta, p, z, T, out);            // z - временный буфер
            #pragma omp parallel for
            for (int i = 0; i < N; ++i) {
                p[i] = z[i];
            }
        }

        // q_k = Ap_k
        spmv(N, IA, JA, A, p, q, T, out);

        // alpha_k = rho_k/(p_k,q_k)
        alpha = rho / dot(N, p, q, T, out);

        // x_k = x_k-1 + alpha_k * p_k
        axpy(N, alpha, p, x, T, out);

        // r_k = r_k-1 - alpha_k * q_k
        axpy(N, -alpha, q, r, T, out);

#ifdef DBG_SOLVER
        // Норма невязки
        double norm = sqrt(dot(N, r, r, T, out));
        printf("%d %e\n", k, norm);
#endif

        rho_prev = rho;

    } while (rho > eps * eps && k < maxit);

    end_phase = omp_get_wtime();
    fprintf(out, "Phase 7 (iterative solution) time: %f seconds\n", end_phase - start_phase);

    // Финальная невязка
    spmv(N, IA, JA, A, x, q, T, out);  // q = Ax
    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        r[i] = q[i] - b[i];    // r = Ax - b
    }
    *res = sqrt(dot(N, r, r, T, out));
    *n = k;

    end_total = omp_get_wtime();
    fprintf(out, "Total Solve execution time: %f seconds\n\n", end_total - start_total);

    free(r);
    free(z);
    free(p);
    free(q);
    free(M);
}
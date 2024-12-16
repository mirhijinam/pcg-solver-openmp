#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include "custom.h"

int Fill(
    int N,
    int* IA,
    int* JA,
    double** A,
    double** b,
    int T,
    FILE* out
) {
    double start_total = omp_get_wtime();

    omp_set_num_threads(T);

    *A = (double*)malloc(IA[N] * sizeof(double));
    if (!*A) return -1;

    *b = (double*)malloc(N * sizeof(double));
    if (!*b) {
        free(*A);
        return -1;
    }

    #pragma omp parallel
    {
        #pragma omp for schedule(static)
        for (int i = 0; i < N; ++i) {
            double sum = 0.0;
            int diag_pos = -1;

            for (int k = IA[i]; k < IA[i + 1]; ++k) {
                int j = JA[k];
                if (i != j) {
                    (*A)[k] = cos(i * j + i + j);
                    sum += fabs((*A)[k]);
                } else {
                    diag_pos = k;
                }
            }

            if (diag_pos >= 0) {
                (*A)[diag_pos] = 1.234 * sum;
            }
        }
    }

    #pragma omp parallel
    {
        #pragma omp for schedule(static)
        for (int i = 0; i < N; ++i) {
            (*b)[i] = sin(i);
        }
    }

    double end_total = omp_get_wtime();
    fprintf(out, "Fill: %e seconds\n", end_total - start_total);

    return 0;
}
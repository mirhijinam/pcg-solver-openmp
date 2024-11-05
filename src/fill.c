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
    int T
) {
    double start_total, end_total, start_phase, end_phase;
    start_total = omp_get_wtime();
    
    omp_set_num_threads(T);

    // Выделяем память под массивы A и b
    *A = (double*)malloc(IA[N] * sizeof(double));
    *b = (double*)malloc(N * sizeof(double));
    
    if (!*A || !*b) {
        if (*A) free(*A);
        if (*b) free(*b);
        return -1;
    }

    // Замер времени первой фазы - заполнение матрицы A
    start_phase = omp_get_wtime();

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i) {
        double sum = 0.0;
        int diag_pos = -1;
        
        // Обрабатываем все элементы в строке i
        for (int k = IA[i]; k < IA[i + 1]; ++k) {
            int j = JA[k];
            
            if (i != j) {
                // Внедиагональный элемент
                (*A)[k] = cos(i * j + i + j);
                sum += fabs((*A)[k]);
            } else {
                // Запоминаем позицию диагонального элемента
                diag_pos = k;
            }
        }
        
        // Заполняем диагональный элемент
        if (diag_pos >= 0) {
            (*A)[diag_pos] = 1.234 * sum;
        }
    }

    end_phase = omp_get_wtime();
    fprintf(stderr, "Phase 4 (filling matrix A) time: %f seconds\n", end_phase - start_phase);

    // Замер времени второй фазы - заполнение вектора b
    start_phase = omp_get_wtime();

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i) {
        (*b)[i] = sin(i);
    }

    end_phase = omp_get_wtime();
    fprintf(stderr, "Phase 5 (filling vector b) time: %f seconds\n", end_phase - start_phase);

#ifdef DBG_FILL
    fprintf(stderr, "Fill results-------------------------------------------------\n");
    fprintf(stderr, "\nA:\n");
    for (int i = 0; i < N; ++i) {
        fprintf(stderr, "---A[%d] = (", i);
        for (int k = IA[i]; k < IA[i + 1]; ++k) {
            fprintf(stderr, " %.6f ", (*A)[k]);
        }
        fprintf(stderr, ")\n");
    }

    fprintf(stderr, "\nb:\n");
    for (int i = 0; i < N; ++i) {
        fprintf(stderr, "---b[%d] = %.6f\n", i, (*b)[i]);
    }
    fprintf(stderr, "-------------------------------------------------------------\n\n");
#endif

    end_total = omp_get_wtime();
    fprintf(stderr, "Total Fill execution time: %f seconds\n\n", end_total - start_total);

    return 0;
}

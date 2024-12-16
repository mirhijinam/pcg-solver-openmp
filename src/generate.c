#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include "custom.h"

int Generate(
    int Nx,
    int Ny,
    int K1,
    int K2,
    int T,
    int** IA,
    int** JA,
    int* N,
    FILE* out
) {
    double start_total, end_total;
    start_total = omp_get_wtime();

    omp_set_num_threads(T);

    *N = (Nx + 1) * (Ny + 1);
    int* neighbors_count = (int*)calloc(*N, sizeof(int));
    if (!neighbors_count) return -1;

    // Подсчёт соседей
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i <= Ny; ++i) {
        for (int j = 0; j <= Nx; ++j) {
            int cur_node = i * (Nx + 1) + j;

            if (i > 0) neighbors_count[cur_node]++; // Верхний
            if (j > 0) neighbors_count[cur_node]++; // Левый
            if (j < Nx) neighbors_count[cur_node]++; // Правый
            if (i < Ny) neighbors_count[cur_node]++; // Нижний

            // Верхний правый по диагонали
            int up_cell_idx = (i - 1) * Nx + j;
            if (i > 0 && j < Nx && up_cell_idx >= 0 && up_cell_idx < Nx * Ny && (up_cell_idx % (K1 + K2)) >= K1) {
                neighbors_count[cur_node]++;
            }

            // Нижний левый по диагонали
            int prev_cell_idx = i * Nx + (j - 1);
            if (i < Ny && j > 0 && prev_cell_idx >= 0 && prev_cell_idx < Nx * Ny && (prev_cell_idx % (K1 + K2)) >= K1) {
                neighbors_count[cur_node]++;
            }
        }
    }

    // Генерация IA
    *IA = (int*)malloc((*N + 1) * sizeof(int));
    if (!*IA) {
        free(neighbors_count);
        return -1;
    }

    (*IA)[0] = 0;
    for (int i = 0; i < *N; ++i) {
        (*IA)[i + 1] = (*IA)[i] + neighbors_count[i];
    }

    // Генерация JA
    *JA = (int*)malloc((*IA)[*N] * sizeof(int));
    if (!*JA) {
        free(neighbors_count);
        free(*IA);
        return -1;
    }

    memset(neighbors_count, 0, *N * sizeof(int));

    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i <= Ny; ++i) {
        for (int j = 0; j <= Nx; ++j) {
            int cur_node = i * (Nx + 1) + j;
            int pos = (*IA)[cur_node];
            int local_count = 0;

            if (i > 0) (*JA)[pos + local_count++] = cur_node - (Nx + 1); // Верхний
            if (j > 0) (*JA)[pos + local_count++] = cur_node - 1;       // Левый
            (*JA)[pos + local_count++] = cur_node;                     // Сам
            if (j < Nx) (*JA)[pos + local_count++] = cur_node + 1;     // Правый
            if (i < Ny) (*JA)[pos + local_count++] = cur_node + (Nx + 1); // Нижний

            // Верхний правый по диагонали
            int up_cell_idx = (i - 1) * Nx + j;
            if (i > 0 && j < Nx && up_cell_idx >= 0 && up_cell_idx < Nx * Ny && (up_cell_idx % (K1 + K2)) >= K1) {
                (*JA)[pos + local_count++] = (i - 1) * (Nx + 1) + (j + 1);
            }

            // Нижний левый по диагонали
            int prev_cell_idx = i * Nx + (j - 1);
            if (i < Ny && j > 0 && prev_cell_idx >= 0 && prev_cell_idx < Nx * Ny && (prev_cell_idx % (K1 + K2)) >= K1) {
                (*JA)[pos + local_count++] = (i + 1) * (Nx + 1) + (j - 1);
            }
        }
    }

    free(neighbors_count);

    end_total = omp_get_wtime();
    fprintf(out, "Generate: %e seconds\n", end_total - start_total);

    return 0;
}
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
    double start_total, end_total, start_phase, end_phase;
    start_total = omp_get_wtime();
    
    omp_set_num_threads(T);
    
    *N = (Nx + 1) * (Ny + 1);
    int* neighbors_count = (int*)calloc(*N, sizeof(int));
    if (!neighbors_count) return -1;

#ifdef DEBUG
    fprintf(out, "Neighbor counter---------------------------------------------\n");
#endif

    // Замер времени первой фазы
    start_phase = omp_get_wtime();

    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i <= Ny; ++i) {
        for (int j = 0; j <= Nx; ++j) {
            int cur_node = i * (Nx + 1) + j;

#ifdef DEBUG
            #pragma omp critical
            {
                fprintf(out, "cur_node=%d\n", cur_node);
            }
#endif

            // Верхний
            if (i > 0) {
                #pragma omp atomic
                neighbors_count[cur_node]++;
#ifdef DEBUG
                #pragma omp critical
                {
                    fprintf(out, "---up_node: %d\n", cur_node - (Nx + 1));
                }
#endif
            }

            // Верхний правый по диагонали
            int up_cell_idx = (i - 1) * Nx + j;
            if (i > 0 && j < Nx && up_cell_idx >= 0 && up_cell_idx < Nx*Ny && (up_cell_idx % (K1 + K2)) >= K1) {
                #pragma omp atomic
                neighbors_count[cur_node]++;
#ifdef DEBUG
                #pragma omp critical
                {
                    fprintf(out, "---up_right_node: %d\n", (i - 1) * (Nx + 1) + (j + 1));
                }
#endif
            }

            // Левый
            if (j > 0) {
                #pragma omp atomic
                neighbors_count[cur_node]++;
#ifdef DEBUG
                #pragma omp critical
                {
                    fprintf(out, "---left_node: %d\n", cur_node - 1);
                }
#endif
            }

            // Сам
            #pragma omp atomic
            neighbors_count[cur_node]++;

            // Правый
            if (j < Nx) {
                #pragma omp atomic
                neighbors_count[cur_node]++;
#ifdef DEBUG
                #pragma omp critical
                {
                    fprintf(out, "---right_node: %d\n", cur_node + 1);
                }
#endif
            }

            // Нижний левый по диагонали
            int prev_cell_idx = i * Nx + (j - 1);
            if (i < Ny && j > 0 && prev_cell_idx >= 0 && prev_cell_idx < Nx*Ny && (prev_cell_idx % (K1 + K2)) >= K1) {
                #pragma omp atomic
                neighbors_count[cur_node]++;
#ifdef DEBUG
                #pragma omp critical
                {
                    fprintf(out, "---down_left_node: %d\n", (i + 1) * (Nx + 1) + (j - 1));
                }
#endif
            }

            // Нижний
            if (i < Ny) {
                #pragma omp atomic
                neighbors_count[cur_node]++;
#ifdef DEBUG
                #pragma omp critical
                {
                    fprintf(out, "---down_node: %d\n", cur_node + (Nx + 1));
                }
#endif
            }
        }
    }

    end_phase = omp_get_wtime();
    fprintf(out, "Phase 1 (counting neighbors) time: %f seconds\n", end_phase - start_phase);

#ifdef DEBUG
    fprintf(out, "-------------------------------------------------------------\n\n");
    fprintf(out, "Number of neighbors------------------------------------------\n");
    #pragma omp parallel for ordered
    for (int i = 0; i < *N; i++) {
        #pragma omp ordered
        fprintf(out, "node %d: %d neighbors\n", i, neighbors_count[i]); 
    }
    fprintf(out, "-------------------------------------------------------------\n\n");
#endif

    // Замер времени второй фазы
    start_phase = omp_get_wtime();

    *IA = (int*)malloc((*N + 1) * sizeof(int));
    if (!*IA) {
        free(neighbors_count);
        return -1;
    }

    (*IA)[0] = 0;
    for (int i = 0; i < *N; ++i) {
        (*IA)[i + 1] = (*IA)[i] + neighbors_count[i];
    }

    end_phase = omp_get_wtime();
    fprintf(out, "Phase 2 (generating IA) time: %f seconds\n", end_phase - start_phase);
    
    *JA = (int*)malloc((*IA)[*N] * sizeof(int));
    if (!*JA) {
        free(neighbors_count);
        free(*IA);
        return -1;
    }

    memset(neighbors_count, 0, *N * sizeof(int));

#ifdef DEBUG
    fprintf(out, "JA-----------------------------------------------------------\n");
#endif

    // Замер времени третьей фазы
    start_phase = omp_get_wtime();

    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i <= Ny; ++i) {
        for (int j = 0; j <= Nx; ++j) {
            int cur_node = i * (Nx + 1) + j;
            int pos = (*IA)[cur_node];
            int local_count = 0;  // локальный счётчик для каждого потока

#ifdef DEBUG
            #pragma omp critical
            {
                fprintf(out, "cur_node=%d, pos=%d\n", cur_node, pos);
            }
#endif

            // Верхний
            if (i > 0) {
                (*JA)[pos + local_count++] = cur_node - (Nx + 1);
#ifdef DEBUG
                #pragma omp critical
                {
                    fprintf(out, "---JA(up): %d\n", cur_node - (Nx + 1));
                }
#endif
            }
            
            // Верхний правый по диагонали
            int up_cell_idx = (i - 1) * Nx + j;
            if (i > 0 && j < Nx && up_cell_idx >= 0 && up_cell_idx < Nx*Ny && (up_cell_idx % (K1 + K2)) >= K1) {
                (*JA)[pos + local_count++] = (i - 1) * (Nx + 1) + (j + 1);
#ifdef DEBUG
                #pragma omp critical
                {
                    fprintf(out, "---JA(up_right): %d\n", (i - 1) * (Nx + 1) + (j + 1));
                }
#endif
            }

            // Левый сосед
            if (j > 0) {
                (*JA)[pos + local_count++] = cur_node - 1;
#ifdef DEBUG
                #pragma omp critical
                {
                    fprintf(out, "---JA(left): %d\n", cur_node - 1);
                }
#endif
            }

            // Сам
            (*JA)[pos + local_count++] = cur_node;

            // Правый сосед
            if (j < Nx) {
                (*JA)[pos + local_count++] = cur_node + 1;
#ifdef DEBUG
                #pragma omp critical
                {
                    fprintf(out, "---JA(right): %d\n", cur_node + 1);
                }
#endif
            }

            // Нижний левый по диагонали
            int prev_cell_idx = i * Nx + (j - 1);
            if (i < Ny && j > 0 && prev_cell_idx >= 0 && prev_cell_idx < Nx*Ny && (prev_cell_idx % (K1 + K2)) >= K1) {
                (*JA)[pos + local_count++] = (i + 1) * (Nx + 1) + (j - 1);
#ifdef DEBUG
                #pragma omp critical
                {
                    fprintf(out, "---JA(down_left): %d\n", (i + 1) * (Nx + 1) + (j - 1));
                }
#endif
            }

            // Нижний
            if (i < Ny) {
                (*JA)[pos + local_count++] = cur_node + (Nx + 1);
#ifdef DEBUG
                #pragma omp critical
                {
                    fprintf(out, "---JA(down): %d\n", cur_node + (Nx + 1));
                }
#endif
            }

#ifdef DEBUG
            #pragma omp critical
            {
                fprintf(out, "------JA: [ ");
                for (int k = pos; k < pos + local_count; k++) {
                    fprintf(out, "%d ", (*JA)[k]);
                }
                fprintf(out, "]\n");
            }
#endif
        }
    }

    end_phase = omp_get_wtime();
    fprintf(out, "Phase 3 (generating JA) time: %f seconds\n", end_phase - start_phase);

#ifdef DEBUG
    fprintf(out, "-------------------------------------------------------------\n\n");
#endif

    free(neighbors_count);

    end_total = omp_get_wtime();
    fprintf(out, "Total Generate execution time: %f seconds\n\n", end_total - start_total);
    
    return 0;
}
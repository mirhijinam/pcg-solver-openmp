#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include "custom.h"

void print_usage(const char* program_name) {
    fprintf(stderr, "usage:\n");
    fprintf(stderr, "%s <Nx> <Ny> <K1> <K2> <T> [-d] [-o output_file]\n", program_name);
    fprintf(stderr, "or\n");
    fprintf(stderr, "%s -f input_file <T> [-d] [-o output_file]\n", program_name);
}

int main(int argc, char* argv[]) {
    int Nx = 0, Ny = 0, K1 = 0, K2 = 0, T = 1;
    int debug_output = 0;
    char* output_file = NULL;
    
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    if (strcmp(argv[1], "-f") == 0) {
        if (argc < 4) {
            fprintf(stderr, "input error: not enough arguments\n");
            return 1;
        }
        FILE* input = fopen(argv[2], "r");
        if (!input) {
            fprintf(stderr, "input error: failed to open file %s\n", argv[2]);
            return 1;
        }
        if (fscanf(input, "%d %d %d %d", &Nx, &Ny, &K1, &K2) != 4) {
            fprintf(stderr, "input error: wrong file format\n");
            fclose(input);
            return 1;
        }
        fclose(input);
        
        T = atoi(argv[3]);
        
        for (int i = 4; i < argc; ++i) {
            if (strcmp(argv[i], "-d") == 0) {
                debug_output = 1;
            } else if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
                output_file = argv[++i];
            }
        }
    } else {
        if (argc < 6) {
            fprintf(stderr, "input error: not enough arguments\n");
            print_usage(argv[0]);
            return 1;
        }
        Nx = atoi(argv[1]);
        Ny = atoi(argv[2]);
        K1 = atoi(argv[3]);
        K2 = atoi(argv[4]);
        T = atoi(argv[5]);

        omp_set_num_threads(T);

        for (int i = 6; i < argc; ++i) {
            if (strcmp(argv[i], "-d") == 0) {
                debug_output = 1;
            } else if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
                output_file = argv[++i];
            }
        }
    }

    if (Nx <= 0 || Ny <= 0 || K1 < 0 || K2 < 0 || T <= 0) {
        fprintf(stderr, "input error: incorrect parameter values\n");
        return 1;
    }

    int N;
    int *IA = NULL, *JA = NULL;
    double *A = NULL, *b = NULL;

    FILE* out = output_file ? fopen(output_file, "w") : stdout;
    fprintf(out, "Nx_T:%d_%d\n\n", Nx, T);
        
    if (output_file && !out) {
        fprintf(stderr, "error: failed to open output file %s\n", output_file);
        return 1;
    }

    int result = Generate(Nx, Ny, K1, K2, T, &IA, &JA, &N, out);
    if (result != 0) {
        fprintf(stderr, "generate error: failed to generate matrix\n");
        return 1;
    }

    result = Fill(N, IA, JA, &A, &b, T, out);
    if (result != 0) {
        fprintf(stderr, "fill error: failed to fill matrix\n");
        free(IA);
        free(JA);
        return 1;
    }

    double* x = (double*)malloc(N * sizeof(double));
    if (!x) {
        fprintf(stderr, "solve error: failed to allocate memory for solution vector\n");
        free(IA);
        free(JA);
        free(A);
        free(b);
        return 1;
    }

    double eps = 1e-5;    // точность
    int maxit = 1000;     // максимальное число итераций
    int iterations;       // фактическое число итераций
    double residual;      // невязка

    Solve(A, b, x, N, IA, JA, eps, maxit, &iterations, &residual, T, out);

    fprintf(out, "\nSolution completed.\n");
    fprintf(out, "\tIterations:%d\n", iterations);
    fprintf(out, "\tResidual:%e\n", residual);

    if (debug_output) {
        fprintf(out, "\nN = %d\n", N);
        fprintf(out, "IA: ");
        for (int i = 0; i <= N; ++i) {
            fprintf(out, "%d ", IA[i]);
        }
        fprintf(out, "\nJA: ");
        for (int i = 0; i < IA[N]; ++i) {
            fprintf(out, "%d ", JA[i]);
        }
        fprintf(out, "\nSolution vector x:\n");
        for (int i = 0; i < N; ++i) {
            fprintf(out, "%e\n", x[i]);
        }
        fprintf(out, "\n");
    }

    if (output_file) {
        fclose(out);
    }

    free(IA);
    free(JA);
    free(A);
    free(b);
    free(x);

    return 0;
}

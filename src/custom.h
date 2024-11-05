#ifndef CUSTOM_H
#define CUSTOM_H

#include <stdio.h>

//#define DBG_GENERATE  // отладка этапа Generate
//#define DBG_FILL      // отладка этапа Fill
//#define DBG_SOLVER    // отладка этапа Solve и кернелов

// Прототипы функций
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
);

int Fill(
    int N,
    int* IA,
    int* JA,
    double** A,
    double** b,
    int T,
    FILE* out
);

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
);

void axpy_seq(
    int N,
    double alpha,
    double* x,
    double* y,
    FILE* out
);

void spmv_seq(
    int N,
    int* IA,
    int* JA,
    double* A,
    double* x,
    double* y,
    FILE* out
);

double dot_seq(
    int N,
    double* x,
    double* y,
    FILE* out
);

void axpy(
    int N,
    double alpha,
    double* x,
    double* y,
    int T,
    FILE* out
);

double dot(
    int N,
    double* x,
    double* y,
    int T,
    FILE* out
);

void spmv(
    int N,
    int* IA,
    int* JA,
    double* A,
    double* x,
    double* y,
    int T,
    FILE* out
);

#endif 
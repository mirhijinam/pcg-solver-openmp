GCC = gcc
CLANG = clang
CFLAGS += -fopenmp

gcc:
	$(GCC) -O3 -Wall -std=c99 -fopenmp -lm src/main.c src/generate.c src/fill.c src/solve.c -o main

clang:
	$(CLANG) -o main src/main.c src/generate.c src/fill.c src/solve.c -I. -Xpreprocessor -fopenmp -I`brew --prefix libomp`/include -L`brew --prefix libomp`/lib -lomp -lm

clean:
	rm -f *.o main
	
GCC = gcc
CLANG = clang
CFLAGS = -fopenmp -march=native -O3 -Wall -std=c99 -lm

gcc:
	$(GCC) $(CFLAGS) src/main.c src/generate.c src/fill.c src/solve.c -o main

clang:
	$(CLANG) -o main src/main.c src/generate.c src/fill.c src/solve.c -I. -Xpreprocessor -fopenmp -I`brew --prefix libomp`/include -L`brew --prefix libomp`/lib -lomp -lm

clean:
	rm -f *.o main
	
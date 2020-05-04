CFLAGS = -Xcompiler -fopenmp -O3
CC = mpic++

all: jacobi	sample_sort

sample_sort: p2_sample_sort.cpp
	${CC} -o sample_sort p2_sample_sort.cpp

jacobi: p1_jacobi.cpp
	${CC} -o jacobi p1_jacobi.cpp
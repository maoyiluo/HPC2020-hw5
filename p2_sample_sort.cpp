#include <stdio.h>
#include <math.h>
#include <mpi.h>
#include <string.h>

int main(int argc, char *argv[])
{
    int mpirank, i;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    /* get name of host running MPI process */
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);
    printf("Rank %d/%d running on %s.\n", mpirank, p, processor_name);

    sscanf(argv[1], "%d", &N);
    sscanf(argv[2], "%d", &max_iters);

    /*generate random local array*/
    long* local_arr = (long*) malloc(sizeof(long)*N);
    for(i = 0; i < N; i++) local_arr[i] = rand() % 1000000;

    long* splitting = (long*) malloc(sizeof(long)*(p-1));


    MPI_Finalize();
    return 0;
}
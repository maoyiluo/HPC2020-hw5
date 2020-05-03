// Parallel sample sort
#include <stdio.h>
#include <unistd.h>
#include <mpi.h>
#include <stdlib.h>
#include <algorithm>

void output_to_file(int rank, int type, int *array, int N)
{
    FILE *fd = NULL;
    char filename[256];
    if (type == 0)
    {
        snprintf(filename, 256, "gather_sample%02d.txt", rank);
    }
    else if (type == 1)
    {
        snprintf(filename, 256, "splitters%02d.txt", rank);
    }
    else
    {
        snprintf(filename, 256, "alltoall%02d.txt", rank);
    }

    fd = fopen(filename, "w+");

    if (NULL == fd)
    {
        printf("Error opening file \n");
    }

    fprintf(fd, "rank %d received the message:\n", rank);
    for (int n = 0; n < N; ++n)
        fprintf(fd, "  %d\n", array[n]);

    fclose(fd);
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int rank, p;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    // Number of random numbers per processor (this should be increased
    // for actual tests or could be passed in through the command line
    int N = 100;
    sscanf(argv[1], "%d", &N);

    int *vec = (int *)malloc(N * sizeof(int));
    // seed random number generator differently on every core
    srand((unsigned int)(rank + 393919));

    // fill vector with random integers
    for (int i = 0; i < N; ++i)
    {
        vec[i] = rand();
    }
    printf("rank: %d, first entry: %d\n", rank, vec[0]);

    // sort locally
    std::sort(vec, vec + N);

    // sample p-1 entries from vector as the local splitters, i.e.,
    // every N/P-th entry of the sorted vector
    int *sample = (int *)malloc(p * sizeof(int));
    for (int i = 0; i < p - 1; i++)
        sample[i] = vec[N / p * i];

    // every process communicates the selected entries to the root
    // process; use for instance an MPI_Gather
    int *gathered_sample;
    if (rank == 0)
        gathered_sample = (int *)malloc(p * (p - 1) * sizeof(int));
    MPI_Barrier(MPI_COMM_WORLD);    
    MPI_Gather(sample, p - 1, MPI_INT, gathered_sample, p - 1, MPI_INT,0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        printf("gathered sample: \n");
        for (int i = 0; i < p * (p - 1); i++)
        {
            printf("%d ", gathered_sample[i]);
        }
        printf("\n");
    }

    // root process does a sort and picks (p-1) splitters (from the
    // p(p-1) received elements)

    int *splitters = (int *)malloc((p - 1) * sizeof(int));
    if (rank == 0)
    {
        std::sort(gathered_sample, gathered_sample + p * (p - 1));
        for (int i = 0; i < p - 1; i++)
        {
            splitters[i] = gathered_sample[i * p];
        }
    }

    // root process broadcasts splitters to all other processes
    MPI_Barrier(MPI_COMM_WORLD); 
    MPI_Bcast(splitters, p - 1, MPI_INT, 0, MPI_COMM_WORLD);

    output_to_file(rank, 1, splitters, p-1);
    // every process uses the obtained splitters to decide which
    // integers need to be sent to which other process (local bins).
    // Note that the vector is already locally sorted and so are the
    // splitters; therefore, we can use std::lower_bound function to
    // determine the bins efficiently.
    //
    // Hint: the MPI_Alltoallv exchange in the next step requires
    // send-counts and send-displacements to each process. Determining the
    // bins for an already sorted array just means to determine these
    // counts and displacements. For a splitter s[i], the corresponding
    // send-displacement for the message to process (i+1) is then given by,
    // sdispls[i+1] = std::lower_bound(vec, vec+N, s[i]) - vec;
    int *send_displacement = (int *)malloc(p * sizeof(int));
    send_displacement[0] = 0;
    for (int i = 0; i < p - 1; i++)
    {
        send_displacement[i + 1] = std::lower_bound(vec, vec + N, splitters[i]) - vec;
    }
    int *receive_displacement = (int *)malloc((p) * sizeof(int));
    // send and receive: first use an MPI_Alltoall to share with every
    // process how many integers it should expect, and then use
    // MPI_Alltoallv to exchange the data
    MPI_Alltoall(send_displacement, p, MPI_INT, receive_displacement, p, MPI_INT, MPI_COMM_WORLD);
     MPI_Barrier(MPI_COMM_WORLD); 
    output_to_file(rank, 2, receive_displacement, p);
    int bucket_size = 0;
    for (int i = 0; i < p; i++)
    {
        bucket_size += receive_displacement[i];
    }
    int *bucket = (int *)malloc(bucket_size * sizeof(int));
    // do a local sort of the received data

    // every process writes its result to a file

    free(vec);
    MPI_Finalize();
    return 0;
}
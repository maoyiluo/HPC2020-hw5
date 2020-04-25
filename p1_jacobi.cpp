/* MPI-parallel Jacobi smoothing to solve -u''=f
 * Global vector has N*N unknowns, each processor works with its
 * part, which has lN = N*N/p unknowns.
 */
#include <stdio.h>
#include <math.h>
#include <mpi.h>
#include <string.h>

/* compuate global residual, assuming ghost values are updated */
double compute_residual(double **lu, int lN, double invhsq)
{
    int i, j;
    double tmp, gres = 0.0, lres = 0.0;

    for (i = 1; i <= lN; i++)
    {
        for (j = 1; j <= lN; j++)
            tmp = ((-lu[i][j - 1] - lu[i][j + 1] + 4.0 * lu[i][j] - lu[i - 1][j] - lu[i + 1][j]) * invhsq - 1);
        lres += tmp * tmp;
    }
    /* use allreduce for convenience; a reduce would also be sufficient */
    MPI_Allreduce(&lres, &gres, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return sqrt(gres);
}

int main(int argc, char *argv[])
{
    int mpirank, i, j, p, N, lN, iter, max_iters, process_per_line;
    MPI_Status left_status, up_status, right_status, bottom_status;

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

    /* compute number of unknowns handled by each process */
    lN = N / sqrt(p);
    process_per_line = sqrt(p);
    if ((N % sqrt(p) != 0) && mpirank == 0)
    {
        printf("N: %d, local N: %d\n", N, lN);
        printf("Exiting. N must be a multiple of p\n");
        MPI_Abort(MPI_COMM_WORLD, 0);
    }
    /* timing */
    MPI_Barrier(MPI_COMM_WORLD);
    double tt = MPI_Wtime();

    /* Allocation of vectors, including left/upper and right/lower ghost points */
    double **lu = (double *)calloc(sizeof(double *), lN + 2);
    double **lunew = (double *)calloc(sizeof(double *), lN + 2);
    double **lutemp;

    // the geomatric informatin of current process.
    int row = mpi_rank / N;
    int col = mpi_rank % N;

    for (i = 0; i < lN + 2; i++)
    {
        lu[i] = (double *)malloc(sizeof(double), lN + 2);
        lunew[i] = (double *)malloc(sizeof(double, lN + 2));
    }

    //initialize the local matrix.
    for (i = 0; i < lN + 2; i++)
    {
        for (j = 0; j < lN + 2; j++)
        {
            lu[i][j] = 1;
            lunew[i][j] = 1;
        }
    }

    //buffer for sending
    double *send_right_boundary = (double *)malloc(sizeof(double), lN);
    double *send_left_boundary = (double *)malloc(sizeof(double), lN);
    double *send_upper_boundary = (double *)malloc(sizeof(double), lN);
    double *send_bottom_boundary = (double *)malloc(sizeof(double), lN);

    //buffer for receiving
    double *receive_right_boundary = (double *)malloc(sizeof(double), lN);
    double *receive_left_boundary = (double *)malloc(sizeof(double), lN);
    double *receive_upper_boundary = (double *)malloc(sizeof(double), lN);
    double *receive_bottom_boundary = (double *)malloc(sizeof(double), lN);


    double h = 1.0 / (N + 1);
    double hsq = h * h;
    double invhsq = 1. / hsq;
    double gres, gres0, tol = 1e-5;

    /* initial residual */
    gres0 = compute_residual(lu, lN, invhsq);
    gres = gres0;

    for (iter = 0; iter < max_iters && gres / gres0 > tol; iter++)
    {

        /* Jacobi step for local points */
        for (i = 1; i <= lN; i++)
        {
            for (j = 1; j < lN; j++)
            {
                lunew[i][j] = 0.25 * (hsq + lu[i][j - 1] + lu[i][j + 1] + lu[i - 1][j] + lu[i + 1][j]);
            }
        }

        /* communicate ghost values */
        if (row > 0)
        {
            /* If not the first row, send/recv bdry values to the process below */
            memcpy(send_bottom_boundary, lunew[1], lN*sizeof(double));
            MPI_Send(send_bottom_boundary, lN, MPI_DOUBLE, mpirank - process_per_line, iter, MPI_COMM_WORLD);
            MPI_Recv(receive_bottom_boundary, lN, MPI_DOUBLE, mpirank - process_per_line, iter, MPI_COMM_WORLD, &bottom_status);
        }
        if (col > 0)
        {
            /* If not the first col, send/recv bdry values to the left */
            for(i = 1; i < lN; i++){
                send_left_boundary[i] = lunew[i][1];
            }
            MPI_Send(send_left_boundary, lN, MPI_DOUBLE, mpirank - 1, iter, MPI_COMM_WORLD);
            MPI_Recv(receive_left_boundary, lN, MPI_DOUBLE, mpirank - 1, iter, MPI_COMM_WORLD, &left_status);
        }
        if(row < process_per_line)
        {
            /*If not the last row, send/recv the up boundary to the process above*/
            memcpy(send_upper_boundary, lunew[lN], lN*sizeof(double));
            MPI_Send(send_upper_boundary, lN, MPI_DOUBLE, mpirank + process_per_line, iter, MPI_COMM_WORLD);
            MPI_Recv(receive_upper_boundary, lN, MPI_DOUBLE, mpirank + process_per_line, iter, MPI_COMM_WORLD);
        }
        if(col < process_per_line){
            for(i = 1; i < lN; i++){
                send_right_boundary[i] = lunew[i][lN];
            }
            MPI_Send(send_right_boundary, lN, MPI_DOUBLE, mpirank + 1, iter, MPI_COMM_WORLD);
            MPI_Recv(receive_right_boundary, lN, MPI_DOUBLE, mpirank + 1, iter, MPI_COMM_WORLD);
        }

        /* use the recevice value to update our lunew*/
        for(i = 1; i < lN; i++){
            // up boundary
            lunew[0][i] = receive_upper_boundary[i];
            // right boundry
            lunew[i][lN+1] = receive_right_boundary[i];
            // left boundary
            lunew[i][0] = receive_left_boundary[i];
            // bottom boundary
            lunew[lN+1][i] = receive_bottom_boundary[i];
        }

        /* copy newu to u using pointer flipping */
        lutemp = lu;
        lu = lunew;
        lunew = lutemp;
        if (0 == (iter % 10))
        {
            gres = compute_residual(lu, lN, invhsq);
            if (0 == mpirank)
            {
                printf("Iter %d: Residual: %g\n", iter, gres);
            }
        }
    }

    /* Clean up */
    free(lu);
    free(lunew);

    /* timing */
    MPI_Barrier(MPI_COMM_WORLD);
    double elapsed = MPI_Wtime() - tt;
    if (0 == mpirank)
    {
        printf("Time elapsed is %f seconds.\n", elapsed);
    }
    MPI_Finalize();
    return 0;
}
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
    process_per_line = sqrt(p);
    lN = N / process_per_line;
    if ((N % process_per_line != 0) && mpirank == 0)
    {
        printf("N: %d, local N: %d\n", N, lN);
        printf("Exiting. N must be a multiple of p\n");
        MPI_Abort(MPI_COMM_WORLD, 0);
    }
    /* timing */
    MPI_Barrier(MPI_COMM_WORLD);
    double tt = MPI_Wtime();

    /* Allocation of vectors, including left/upper and right/lower ghost points */
    double **lu = (double **)calloc(sizeof(double *), lN + 2);
    double **lunew = (double **)calloc(sizeof(double *), lN + 2);
    double **lutemp;

    // the geomatric informatin of current process.
    int process_per_row = sqrt(p);
    int row = mpirank / process_per_row;
    int col = mpirank % process_per_row;


    for (i = 0; i < lN + 2; i++)
    {
        lu[i] = (double *)malloc(sizeof(double) * (lN + 2));
        lunew[i] = (double *)malloc(sizeof(double) * (lN + 2));
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
    double *send_right_boundary = (double *)malloc(sizeof(double) * lN);
    double *send_left_boundary = (double *)malloc(sizeof(double) * lN);
    double *send_upper_boundary = (double *)malloc(sizeof(double) * lN);
    double *send_bottom_boundary = (double *)malloc(sizeof(double) * lN);

    //buffer for receiving
    double *receive_right_boundary = (double *)malloc(sizeof(double) * lN);
    double *receive_left_boundary = (double *)malloc(sizeof(double) * lN);
    double *receive_upper_boundary = (double *)malloc(sizeof(double) * lN);
    double *receive_bottom_boundary = (double *)malloc(sizeof(double) * lN);
    
    //request for receive quest
    MPI_Request *receive_right_quest =(MPI_Request*) malloc(sizeof(MPI_Request)); 
    MPI_Request *receive_left_quest =(MPI_Request*) malloc(sizeof(MPI_Request)); 
    MPI_Request *receive_bottom_quest =(MPI_Request*) malloc(sizeof(MPI_Request)); 
    MPI_Request *receive_up_quest =(MPI_Request*) malloc(sizeof(MPI_Request)); 

    //request for send quest
    MPI_Request *send_right_quest =(MPI_Request*) malloc(sizeof(MPI_Request)); 
    MPI_Request *send_left_quest =(MPI_Request*) malloc(sizeof(MPI_Request)); 
    MPI_Request *send_bottom_quest =(MPI_Request*) malloc(sizeof(MPI_Request)); 
    MPI_Request *send_up_quest =(MPI_Request*) malloc(sizeof(MPI_Request)); 


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
                lunew[i][j] = 0.25 * (-hsq + lu[i][j - 1] + lu[i][j + 1] + lu[i - 1][j] + lu[i + 1][j]);
            }
        }

        /* communicate ghost values */
        if (row > 0)
        {
            /* If not the first row, send/recv bdry values to the process below */
            memcpy(send_bottom_boundary, lunew[1], lN*sizeof(double));
            MPI_Isend(send_bottom_boundary, lN, MPI_DOUBLE, mpirank - process_per_line, iter, MPI_COMM_WORLD, send_bottom_quest);
            MPI_Irecv(receive_bottom_boundary, lN, MPI_DOUBLE, mpirank - process_per_line, iter, MPI_COMM_WORLD, receive_bottom_quest);
        }
        if (col > 0)
        {
            /* If not the first col, send/recv bdry values to the left */
            for(i = 1; i < lN; i++){
                send_left_boundary[i] = lunew[i][1];
            }
            MPI_Isend(send_left_boundary, lN, MPI_DOUBLE, mpirank - 1, iter, MPI_COMM_WORLD, send_left_quest);
            MPI_Irecv(receive_left_boundary, lN, MPI_DOUBLE, mpirank - 1, iter, MPI_COMM_WORLD, receive_left_quest);
        }
        if(row < process_per_line - 1)
        {
            /*If not the last row, send/recv the up boundary to the process above*/
            memcpy(send_upper_boundary, lunew[lN], lN*sizeof(double));
            MPI_Isend(send_upper_boundary, lN, MPI_DOUBLE, mpirank + process_per_line, iter, MPI_COMM_WORLD, send_up_quest);
            MPI_Irecv(receive_upper_boundary, lN, MPI_DOUBLE, mpirank + process_per_line, iter, MPI_COMM_WORLD, receive_up_quest);
        }
        if(col < process_per_line - 1){
            for(i = 1; i < lN; i++){
                send_right_boundary[i] = lunew[i][lN];
            }
            MPI_Isend(send_right_boundary, lN, MPI_DOUBLE, mpirank + 1, iter, MPI_COMM_WORLD, send_right_quest);
            MPI_Irecv(receive_right_boundary, lN, MPI_DOUBLE, mpirank + 1, iter, MPI_COMM_WORLD, receive_right_quest);
        }

        /*wait for the updated data*/
        if(row > 0) MPI_Wait(receive_bottom_quest, &bottom_status);
        if(row < process_per_line - 1) MPI_Wait(receive_up_quest, &up_status);
        if(col > 0) MPI_Wait(receive_left_quest, &left_status);
        if(col < process_per_line - 1) MPI_Wait(receive_right_quest, &right_status);

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
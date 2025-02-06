#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#define N 4

// Chatgpt for basic understanding of the algorithm and pseudocode
// MPI documentation for method details like MPI_Bcast, MPI_scatter etc..
// Youtube for visual understanding of the algorithm
// Claude.ai to ask queries and difficulties faced while coding

int main(int argc, char *argv[])
{

  int my_col, my_row;
  double my_a = 0;
  double my_b = 0;
  double my_c = 0;
  int rank, size;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int q = (int)sqrt(size);
  if (q * q != size)
  {
    if (rank == 0)
    {
      printf("Number of processes must be a perfect square\n");
    }
    MPI_Finalize();
    return 1;
  }

  double *matrix_a = (double *)malloc(N * N * sizeof(double));
  double *matrix_b = (double *)malloc(N * N * sizeof(double));

  int dims[2] = {q, q};
  int periods[2] = {1, 1};

  MPI_Comm grid_comm;

  MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &grid_comm);

  int coords[2];
  MPI_Cart_coords(grid_comm, rank, 2, coords);
  my_row = coords[0];
  my_col = coords[1];

  if (rank == 0)
  {
    for (int i = 0; i < N * N; i++)
    {

      matrix_a[i] = (double)(i + 1);
      matrix_b[i] = (double)(i + 5);
    }
  }

  MPI_Scatter(matrix_a, 1, MPI_DOUBLE, &my_a, 1, MPI_DOUBLE, 0, grid_comm);
  MPI_Scatter(matrix_b, 1, MPI_DOUBLE, &my_b, 1, MPI_DOUBLE, 0, grid_comm);

  MPI_Comm row_comm;
  int remain_dims[2] = {0, 1};

  MPI_Cart_sub(grid_comm, remain_dims, &row_comm);

  for (int stage = 0; stage < q; stage++)
  {
    double temp_a = 0;

    if (my_col == (my_row + stage) % q)
    {
      temp_a = my_a;
    }

    MPI_Bcast(&temp_a, 1, MPI_DOUBLE, (my_row + stage) % q, row_comm);

    my_c += temp_a * my_b;

    int src, dest;
    MPI_Cart_shift(grid_comm, 0, -1, &src, &dest);

    printf("Rank %d: Before Sendrecv_replace - my_row=%d, my_col=%d, dest=%d, src=%d, my_b=%f\n",
           rank, my_row, my_col, dest, src, my_b);
    MPI_Sendrecv_replace(&my_b, 1, MPI_DOUBLE, dest, stage, src, stage, grid_comm, MPI_STATUS_IGNORE);
    printf("Rank %d: After Sendrecv_replace - new my_b=%f\n", rank, my_b);
  }

  double *matrix_c = NULL;
  if (rank == 0)
  {
    printf("Initialized matrix c");
    matrix_c = (double *)malloc(N * N * sizeof(double));
  }

  printf("Rank %d: Before Gather\n", rank);
  MPI_Gather(&my_c, 1, MPI_DOUBLE, matrix_c, 1, MPI_DOUBLE, 0, grid_comm);
  printf("Rank %d: After Gather\n", rank);

  MPI_Barrier(MPI_COMM_WORLD);

  if (rank == 0)
  {
    printf("Result matrix C:\n");
    for (int i = 0; i < N; i++)
    {
      for (int j = 0; j < N; j++)
      {
        printf("%.1f ", matrix_c[i * N + j]);
      }
      printf("\n");
    }

    free(matrix_a);
    free(matrix_b);
    free(matrix_c);
  }

  MPI_Comm_free(&row_comm);
  MPI_Comm_free(&grid_comm);
  MPI_Finalize();
  return 0;
}

// // mpicc fox-algo-task-a.c -o fox-algo-task-a
// // mpirun -n 4 --oversubscribe fox-algo-task-a

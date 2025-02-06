#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <mpi.h>

#define N 256

void multiply_submatrices(double *sub_a, double *sub_b, double *sub_c, int block_size)
{
  for (int i = 0; i < block_size; i++)
  {
    for (int j = 0; j < block_size; j++)
    {
      for (int k = 0; k < block_size; k++)
      {
        sub_c[i * block_size + j] +=
            sub_a[i * block_size + k] * sub_b[k * block_size + j];
      }
    }
  }
}

int main(int argc, char *argv[])
{
  int my_col, my_row;
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

  int block_size = N / q;
  if (N % q != 0)
  {
    if (rank == 0)
    {
      printf("Matrix size must be divisible by sqrt(number of processes)\n");
    }
    MPI_Finalize();
    return 1;
  }

  double *my_a = (double *)malloc(block_size * block_size * sizeof(double));
  double *my_b = (double *)malloc(block_size * block_size * sizeof(double));
  double *my_c = (double *)malloc(block_size * block_size * sizeof(double));
  double *temp_a = (double *)malloc(block_size * block_size * sizeof(double));

  for (int i = 0; i < block_size * block_size; i++)
  {
    my_c[i] = 0.0;
  }

  int dims[2] = {q, q};
  int periods[2] = {1, 1};
  MPI_Comm grid_comm;
  MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &grid_comm);

  int coords[2];
  MPI_Cart_coords(grid_comm, rank, 2, coords);
  my_row = coords[0];
  my_col = coords[1];

  double *matrix_a = NULL, *matrix_b = NULL;
  if (rank == 0)
  {
    matrix_a = (double *)malloc(N * N * sizeof(double));
    matrix_b = (double *)malloc(N * N * sizeof(double));
    for (int i = 0; i < N * N; i++)
    {
      matrix_a[i] = (double)(i + 1);
      matrix_b[i] = (double)(i + 5);
    }
  }

  MPI_Datatype block_type;
  MPI_Datatype temp_type;
  MPI_Type_vector(block_size, block_size, N, MPI_DOUBLE, &temp_type);
  MPI_Type_create_resized(temp_type, 0, sizeof(double), &block_type);
  MPI_Type_commit(&block_type);

  int *sendcounts = NULL, *displs = NULL;
  if (rank == 0)
  {
    sendcounts = (int *)malloc(size * sizeof(int));
    displs = (int *)malloc(size * sizeof(int));
    for (int i = 0; i < q; i++)
    {
      for (int j = 0; j < q; j++)
      {
        sendcounts[i * q + j] = 1;
        displs[i * q + j] = i * N * block_size + j * block_size;
      }
    }
  }

  MPI_Scatterv(matrix_a, sendcounts, displs, block_type,
               my_a, block_size * block_size, MPI_DOUBLE, 0, grid_comm);
  MPI_Scatterv(matrix_b, sendcounts, displs, block_type,
               my_b, block_size * block_size, MPI_DOUBLE, 0, grid_comm);

  MPI_Comm row_comm;
  int remain_dims[2] = {0, 1};
  MPI_Cart_sub(grid_comm, remain_dims, &row_comm);

  // Main Fox algorithm loop
  for (int stage = 0; stage < q; stage++)
  {
    // Broadcast A blocks along rows
    if (my_col == (my_row + stage) % q)
    {
      memcpy(temp_a, my_a, block_size * block_size * sizeof(double));
    }
    MPI_Bcast(temp_a, block_size * block_size, MPI_DOUBLE,
              (my_row + stage) % q, row_comm);

    multiply_submatrices(temp_a, my_b, my_c, block_size);

    int src, dest;
    MPI_Cart_shift(grid_comm, 0, -1, &src, &dest);
    MPI_Sendrecv_replace(my_b, block_size * block_size, MPI_DOUBLE,
                         dest, 0, src, 0, grid_comm, MPI_STATUS_IGNORE);
  }

  double *matrix_c = NULL;
  if (rank == 0)
  {
    matrix_c = (double *)malloc(N * N * sizeof(double));
  }

  MPI_Gatherv(my_c, block_size * block_size, MPI_DOUBLE,
              matrix_c, sendcounts, displs, block_type, 0, grid_comm);

  if (rank == 0)
  {
    printf("Result matrix C:\n");
    for (int i = 0; i < 10; i++)
    {
      for (int j = 0; j < 10; j++)
      {
        printf("%.1f ", matrix_c[i * N + j]);
      }
      printf("\n");
    }

    free(matrix_a);
    free(matrix_b);
    free(matrix_c);
    free(sendcounts);
    free(displs);
  }

  free(my_a);
  free(my_b);
  free(my_c);
  free(temp_a);
  MPI_Type_free(&block_type);
  MPI_Comm_free(&row_comm);
  MPI_Comm_free(&grid_comm);
  MPI_Finalize();
  return 0;
}
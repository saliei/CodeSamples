#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>

/*
* proc root sends a token around a ring, each proc
* sends it to the next one. program terminates
* when the root proc gets the token back.
*/
int main(int argc, char** argv)
{
	int rank, size;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	int root = 0;
	int tag = 0;
	int next = (rank + 1) % size;
	int prev = (rank + size - 1) % size;
	int token;
	MPI_Status status;
	/*
	* every proc except the root should wait for the
	* token to be receive, every proc should send the
	* token to next proc, at the end root should recv
	* it back from prev proc. 
	*/
	if(rank != root)
	{
		MPI_Recv(&token, 1, MPI_INT, prev, tag, MPI_COMM_WORLD, &status);
		printf("proc %d received %d from %d.\n", rank, token, prev);
	}
	else
	{
		token = 100;
	}

	MPI_Send(&token, 1, MPI_INT, next, tag, MPI_COMM_WORLD);
	printf("proc %d sent %d to %d.\n", rank, token, next);

	/*
	* this step is needed to avoid deadlock
	*/
	if(rank == root)
	{
		MPI_Recv(&token, 1, MPI_INT, prev, tag, MPI_COMM_WORLD, &status);
		printf("END: proc % d received %d from %d.\n", rank, token, prev);
	}

	MPI_Finalize();
	return MPI_SUCCESS;
}
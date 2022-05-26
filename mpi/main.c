#include "mpi.h"
 
int main(int argc, char *argv[])
{
/* It's important to put this call at the begining of the program, after variable declarations. */
MPI_Init(argc, argv);
 
/* Get the number of MPI processes and the rank of this process. */
	int myRank, numProcs;

	MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
        MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
 
// ==== Call function 'call_me_maybe' from CUDA file multiply.cu: ==========
call_me_maybe();
 
/* ... */
 
}

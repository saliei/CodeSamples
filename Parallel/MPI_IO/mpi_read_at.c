#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>

#define COUNT 1000000

#define CHECK_ERR(func){ \
    if( err != MPI_SUCCESS  ){ \
        int errStringLen; \
        char errString[MPI_MAX_ERROR_STRING]; \
        MPI_Error_string(err, errString, &errStringLen); \
        printf("Error at line %d: calling %s (%s) \n", __LINE__, #func, errString); \
    } \
}

int main(int argc, char** argv){
    int rank, size, err, amode, inints, i;
    int count;
    int buf_w[COUNT];
//    int buf_r[COUNT];
    char* filename;
   
    MPI_File fh; 
    MPI_Comm comm      = MPI_COMM_WORLD;
    MPI_Info info      = MPI_INFO_NULL;
    MPI_Status *status = MPI_STATUS_IGNORE;
    MPI_Datatype etype = MPI_INT;

    MPI_Init(&argc, &argv);
    err = MPI_Comm_rank(comm, &rank);
    CHECK_ERR( MPI_Comm_rank );
    err = MPI_Comm_size(comm, &size);
    CHECK_ERR( MPI_Comm_size );

    filename = "test_mpi_io.dat";
    if( argc > 1 )
        filename = argv[1];

    amode  = MPI_MODE_RDONLY;
    
    err = MPI_File_open(comm, filename, amode, info, &fh);
    CHECK_ERR( MPI_File_open to read );

    int FILESIZE = 4000000; 
    int nints = FILESIZE / (size * sizeof(int));
    int offset = rank * nints * sizeof(int);
    int buf_r[nints];

    MPI_File_read_at(fh, offset, buf_r, nints, etype, status);

    if( rank == 0 ) printf("buf_r[10]: %d", buf_r[nints-1]);
    
    MPI_Get_count(status, etype, &count);
    printf("Process %d: Reads count = %d Bytes.\n", rank, count);
    MPI_File_close(&fh);


program_exit:
    MPI_Finalize();
    return 0;


}
